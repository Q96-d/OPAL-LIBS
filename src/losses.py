# src/losses.py

import tensorflow as tf


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """
    监督对比损失 (Supervised Contrastive Loss)

    参数:
        features: L2归一化后的特征向量，shape (batch_size, feature_dim)
        labels:   标签，shape (batch_size,)
        temperature: 温度系数，控制Softmax分布的锐度

    返回:
        loss: 标量张量
    """
    batch_size = tf.shape(features)[0]

    # 计算相似度矩阵 (点积，因为特征已L2归一化)
    sim_matrix = tf.matmul(features, features, transpose_b=True) / temperature

    # 构建正样本掩码 (同类为1，异类为0)
    labels = tf.reshape(labels, (batch_size, 1))
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
    mask = mask - tf.eye(batch_size)  # 排除自身

    # 计算分子：正样本对的指数和
    exp_sim = tf.exp(sim_matrix)
    pos_exp = tf.reduce_sum(exp_sim * mask, axis=1)

    # 计算分母：所有样本对的指数和（排除自身）
    neg_mask = 1.0 - tf.eye(batch_size)
    neg_exp = tf.reduce_sum(exp_sim * neg_mask, axis=1)

    # 损失：对每个样本，最大化正样本的相对相似度
    loss = -tf.reduce_mean(tf.math.log(pos_exp / (neg_exp + 1e-8)))

    return loss


# ═══════════════════════════════════════════════════════════════
# ArcFace 边距损失（Additive Angular Margin Loss）
# ═══════════════════════════════════════════════════════════════

def arcface_loss(features, labels, kernel, num_classes, s=30.0, m=0.50):
    """
    ArcFace 损失：在角度空间添加加性边距 (additive angular margin)，
    强制同类特征聚集、异类特征远离，从而同步提升开集识别与增量学习性能。

    核心公式:
        L = -log( exp(s * cos(θ_y + m)) / Σ_j exp(s * cos(θ_j + m·δ_{j=y})) )

    其中:
        cos(θ_j) = (W_j^T · x) / (||W_j|| · ||x||)  即归一化特征与归一化权重的点积
        δ_{j=y} = 1 当 j=y（正确类），否则 0（只对正确类施加 margin）

    参数:
        features:     L2 归一化后的特征向量，shape (batch_size, feature_dim)
        labels:       标签，shape (batch_size,)，值域 [0, num_classes)
        kernel:       分类器权重矩阵，shape (feature_dim, num_classes)
        num_classes:  类别总数
        s:            缩放因子 (scale)，控制 softmax 的动态范围，默认 30.0
        m:            角度边距 (margin)，单位弧度，默认 0.50 (~28.6°)

    返回:
        loss: 标量张量

    注意:
        features 和 kernel 必须事先各自 L2 归一化，使得点积等于 cosθ。
    """
    # 权重 L2 归一化（沿特征维度），得到单位超球面上的类锚点
    kernel_norm = tf.math.l2_normalize(kernel, axis=0)  # (dim, num_classes)

    # cosθ_j = 归一化特征 · 归一化权重，shape (batch, num_classes)
    cos_theta = tf.matmul(features, kernel_norm)

    # 数值裁剪，防止 acos → sqrt(1 - cos²) 出现 NaN
    cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

    # 构建 one-hot mask：正确类位置为 1，其余为 0
    batch_size = tf.shape(features)[0]
    mask = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)

    # 三角函数：cos(θ + m) = cosθ·cos(m) - sinθ·sin(m)
    sin_theta = tf.sqrt(1.0 - cos_theta ** 2)
    cos_m = tf.cos(m)
    sin_m = tf.sin(m)
    cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

    # 只对正确类施加边距，其他类保持原 cosθ
    cos_theta_with_margin = mask * cos_theta_m + (1.0 - mask) * cos_theta

    # 缩放并计算 softmax 交叉熵
    logits = s * cos_theta_with_margin
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )

    return loss


def arcface_logits(features, kernel, num_classes, s=30.0):
    """
    计算 ArcFace 风格的推理 logits（无 margin，仅归一化 + 缩放）。

    训练时用 arcface_loss（带 margin），推理时用此函数计算 s·cosθ，
    保证训练/推理的 logits 量级一致（均基于归一化特征空间）。

    参数:
        features:     L2 归一化后的特征向量，shape (batch_size, feature_dim)
        kernel:       分类器权重矩阵，shape (feature_dim, num_classes)
        num_classes:  类别总数
        s:            缩放因子，应与训练时一致，默认 30.0

    返回:
        logits: shape (batch_size, num_classes)，可直接用于 softmax/argmax
    """
    kernel_norm = tf.math.l2_normalize(kernel, axis=0)
    cos_theta = tf.matmul(features, kernel_norm)
    return s * cos_theta
