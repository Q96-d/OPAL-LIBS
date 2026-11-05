# src/incremental.py

import numpy as np
import tensorflow as tf
from src.config import (
    RANDOM_SEED,
    EXEMPLAR_SIZE, EXEMPLAR_BOUNDARY_RATIO,
    EPOCHS_INCREMENTAL, INC_BATCH_SIZE,
    INC_LEARNING_RATE, DISTILLATION_ALPHA, INC_WEIGHT_DECAY,
    INC_SUPCON_CLASSES_PER_BATCH, INC_SUPCON_SAMPLES_PER_CLASS,
    INC_SUPCON_LAMBDA, SUPCON_TEMPERATURE,
    ARCFACE_S, ARCFACE_M
)
from src.losses import supervised_contrastive_loss, arcface_loss

# SupCon 每类最少样本数：至少需要 2 条构成一对正样本对，
# 推荐 ≥4 以保证充足的梯度信号（C(n,2) 对，n≥4 时 ≥6 对）。
MIN_SUPCON_SAMPLES_PER_CLASS = 2

class IncrementalLearner:
    def __init__(self, feature_extractor, classifier):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.exemplars = {}          # {mapped_label: (raw_data_with_channel, mapped_labels)}
        self.exemplar_features = None
        self.exemplar_labels = None
        self.teacher_feature_extractor = None

    def select_exemplars(self, X, y, class_id, m=EXEMPLAR_SIZE,
                          boundary_ratio=None):
        """
        混合范例选择：中心样本 + 边界样本。

        中心样本：离类中心最近的，代表类的"典型"形态
        边界样本：离类中心较远但在合理范围内的，代表类的"分布边界"

        边界样本对 OSR 至关重要——只用中心范例会导致阈值估算偏窄，
        增量后已知类样本容易被误判为未知。

        参数：
            boundary_ratio: 边界样本占比。None 则使用 EXEMPLAR_BOUNDARY_RATIO，
                           设为 0 恢复纯中心选择。
        """
        if boundary_ratio is None:
            boundary_ratio = EXEMPLAR_BOUNDARY_RATIO

        if X.ndim == 2:
            X = X[..., np.newaxis]
        features = self.feature_extractor.predict(X)
        center = features.mean(axis=0)
        distances = np.linalg.norm(features - center, axis=1)

        n_boundary = max(1, int(m * boundary_ratio))
        n_center = m - n_boundary

        idx_sorted = np.argsort(distances)

        # 中心范例：离中心最近的
        center_idx = idx_sorted[:n_center]

        # 边界范例：在合理范围内但离中心最远的
        # 用 95% 分位数排除极端离群点
        p95 = np.percentile(distances, 95)
        boundary_mask = distances <= p95
        boundary_candidates = np.where(boundary_mask)[0]
        if len(boundary_candidates) > n_center:
            # 从边界候选中选最远的 n_boundary 个
            boundary_dist_order = np.argsort(distances[boundary_candidates])
            boundary_idx = boundary_candidates[
                boundary_dist_order[-n_boundary:]
            ]
        else:
            # 样本太少，退化为选最远的
            boundary_idx = idx_sorted[-n_boundary:]

        selected = np.concatenate([center_idx, boundary_idx])
        self.exemplars[class_id] = (X[selected], y[selected])
        self._update_exemplar_buffer()

    def _update_exemplar_buffer(self):
        """将各类范例合并为训练缓冲（原始数据 + 映射后标签）"""
        all_data = []
        all_labels = []
        for data, labels in self.exemplars.values():
            all_data.append(data)
            all_labels.append(labels)
        if all_data:
            self.exemplar_features = np.concatenate(all_data, axis=0)  # 原始数据（含通道维）
            self.exemplar_labels = np.concatenate(all_labels, axis=0)  # 映射后标签

    def save_teacher(self):
        """克隆当前特征提取器作为教师（用于关系蒸馏）"""
        self.teacher_feature_extractor = tf.keras.models.clone_model(self.feature_extractor)
        self.teacher_feature_extractor.set_weights(self.feature_extractor.get_weights())

    def relation_distillation_loss(self, student_features, teacher_features):
        """计算 L2 归一化后距离矩阵的 MSE（关系蒸馏）"""
        t_feats = tf.constant(teacher_features, dtype=tf.float32)
        s_feats = student_features

        # L2 归一化，消除特征范数差异对距离矩阵的干扰
        t_feats = tf.math.l2_normalize(t_feats, axis=1)
        s_feats = tf.math.l2_normalize(s_feats, axis=1)

        t_dist = tf.norm(t_feats[:, None] - t_feats[None, :], axis=-1)
        s_dist = tf.norm(s_feats[:, None] - s_feats[None, :], axis=-1)
        return tf.reduce_mean(tf.square(s_dist - t_dist))

    def incremental_train(self, X_new, y_new_mapped,
                           epochs=None, lr=None, alpha=None,
                           batch_size=None, weight_decay=None,
                           projection_head=None,
                           supcon_lambda=None,
                           supcon_temperature=None):
        """
        增量训练：新类数据 + 范例集回放 + 知识蒸馏 + 监督对比。

        总损失: L_total = L_CE + α·L_Distill + β·L_SupCon

        参数：
            X_new: 新类原始数据，shape (n, dim, 1)
            y_new_mapped: 新类映射后标签，值域 0 ~ K-1
            epochs: 训练轮数
            lr: 学习率
            alpha: 蒸馏损失权重
            batch_size: (未使用，由 SupCon 采样策略覆盖)
            weight_decay: 权重衰减
            projection_head: SupCon 投影头 Keras Model
            supcon_lambda: SupCon 权重 (β)，None 则使用 config 默认值
            supcon_temperature: 温度系数
        """
        # 参数默认值
        epochs = epochs if epochs is not None else EPOCHS_INCREMENTAL
        lr = lr if lr is not None else INC_LEARNING_RATE
        alpha = alpha if alpha is not None else DISTILLATION_ALPHA
        weight_decay = weight_decay if weight_decay is not None else INC_WEIGHT_DECAY
        supcon_lambda = supcon_lambda if supcon_lambda is not None else INC_SUPCON_LAMBDA
        supcon_temperature = supcon_temperature if supcon_temperature is not None else SUPCON_TEMPERATURE

        use_supcon = (projection_head is not None and supcon_lambda > 0)

        # ---- 准备训练数据 ----
        X_new_list = list(X_new)
        y_new_list = list(y_new_mapped)
        has_exemplars = self.exemplar_features is not None
        if has_exemplars:
            X_new_list.extend(self.exemplar_features)
            y_new_list.extend(self.exemplar_labels)
        X_train = np.array(X_new_list, dtype=np.float32)
        y_train = np.array(y_new_list, dtype=np.int64)

        # ---- 打印数据分布 ----
        all_labels, all_counts = np.unique(y_train, return_counts=True)
        new_class_labels = np.unique(y_new_mapped)
        new_class_mask = np.isin(all_labels, new_class_labels)
        old_class_mask = ~new_class_mask

        n_new_total = all_counts[new_class_mask].sum() if new_class_mask.any() else 0
        n_old_total = all_counts[old_class_mask].sum() if old_class_mask.any() else 0

        print(f"  [数据分布]")
        print(f"    训练集中新类样本数量:   {len(y_new_mapped)}")
        print(f"    训练集中旧范例数量:     {n_old_total}")
        print(f"    混合后总样本:           {len(y_train)}  (新类 {n_new_total} + 旧范例 {n_old_total})")
        print(f"    新类 ({new_class_mask.sum()} 类): "
              f"每类 {all_counts[new_class_mask].min()}~{all_counts[new_class_mask].max()} 样本, "
              f"均值 {all_counts[new_class_mask].mean():.1f}")
        if old_class_mask.sum() > 0:
            print(f"    旧类范例 ({old_class_mask.sum()} 类): "
                  f"每类 {all_counts[old_class_mask].min()}~{all_counts[old_class_mask].max()} 样本, "
                  f"均值 {all_counts[old_class_mask].mean():.1f}")
        else:
            print(f"    旧类范例: 无")

        if use_supcon:
            unique_classes = np.unique(y_train)
            n_classes = len(unique_classes)
            classes_per_batch = min(INC_SUPCON_CLASSES_PER_BATCH, n_classes)
            samples_per_class = INC_SUPCON_SAMPLES_PER_CLASS
            class_indices = {c: np.where(y_train == c)[0] for c in unique_classes}

            # ── 采样参数验证 ──
            if samples_per_class < MIN_SUPCON_SAMPLES_PER_CLASS:
                raise ValueError(
                    f"INC_SUPCON_SAMPLES_PER_CLASS={samples_per_class} "
                    f"低于 SupCon 最低要求 ({MIN_SUPCON_SAMPLES_PER_CLASS})，"
                    f"无法构成正样本对。"
                )
            if samples_per_class < 4:
                print(f"  ⚠ [警告] INC_SUPCON_SAMPLES_PER_CLASS={samples_per_class} < 4，"
                      f"正样本对数量仅 C({samples_per_class},2)={samples_per_class*(samples_per_class-1)//2}，"
                      f"建议设为 ≥4。")

            # 检查每类可用样本数
            min_available = min(len(class_indices[c]) for c in unique_classes)
            if min_available < samples_per_class:
                print(f"  ⚠ [警告] 部分类别仅有 {min_available} 条样本 "
                      f"（< 要求的 {samples_per_class}），将使用有放回采样补足。")

            batch_size_actual = classes_per_batch * samples_per_class
            print(f"  [SupCon] β={supcon_lambda}, τ={supcon_temperature}, "
                  f"采样={classes_per_batch}类×{samples_per_class}样本={batch_size_actual}")

        # 保存教师模型
        self.save_teacher()

        # 编译优化器
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            weight_decay=weight_decay
        )

        # ---- 构建训练数据集 ----
        if use_supcon:
            # 类别平衡 + 新旧混合采样
            samples_per_class_actual = samples_per_class

            def balanced_generator():
                rng = np.random.RandomState(RANDOM_SEED)  # 本地 RNG，隔离全局状态
                while True:
                    chosen = rng.choice(
                        unique_classes, classes_per_batch, replace=False
                    )
                    batch_indices = []
                    for c in chosen:
                        idx_pool = class_indices[c]
                        n_available = len(idx_pool)
                        # 优先无放回采样；若样本不足则使用有放回采样保证每类至少
                        # samples_per_class 条，确保 SupCon 获得足够的正样本梯度信号
                        if n_available >= samples_per_class_actual:
                            idx = rng.choice(idx_pool, samples_per_class_actual, replace=False)
                        else:
                            idx = rng.choice(idx_pool, samples_per_class_actual, replace=True)
                        batch_indices.extend(idx)
                    rng.shuffle(batch_indices)
                    yield X_train[batch_indices], y_train[batch_indices]

            train_dataset = tf.data.Dataset.from_generator(
                balanced_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, X_train.shape[1], 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.int64),
                )
            ).prefetch(tf.data.AUTOTUNE)

            steps_per_epoch = max(len(y_train) // batch_size_actual, 1)
        else:
            # 简单随机打乱
            rng = np.random.RandomState(RANDOM_SEED)
            indices = rng.permutation(len(y_train))
            X_shuffled = tf.gather(X_train, indices)
            y_shuffled = tf.gather(y_train, indices)
            dataset = tf.data.Dataset.from_tensor_slices(
                (X_shuffled, y_shuffled)
            ).batch(batch_size or INC_BATCH_SIZE)
            train_dataset = dataset
            steps_per_epoch = None  # 遍历全部数据

        # ---- 训练循环 ----
        for epoch in range(epochs):
            batches = train_dataset
            if steps_per_epoch is not None:
                batches = batches.take(steps_per_epoch)

            for _batch_i, (X_batch, y_batch) in enumerate(batches):
                with tf.GradientTape() as tape:
                    # 学生特征
                    student_feats = self.feature_extractor(X_batch, training=True)

                    # ---- ArcFace Loss（替代 CE）----
                    # L2 归一化特征，使点积等于 cosθ
                    student_feats_norm = tf.math.l2_normalize(student_feats, axis=1)
                    loss_arcface = arcface_loss(
                        student_feats_norm, y_batch,
                        self.classifier.get_kernel(),
                        self.classifier.num_classes,
                        s=ARCFACE_S, m=ARCFACE_M
                    )

                    # ---- 蒸馏 Loss（关系）----
                    loss_distill = tf.constant(0.0)
                    if self.teacher_feature_extractor is not None:
                        teacher_batch_feats = self.teacher_feature_extractor(
                            X_batch, training=False
                        )
                        loss_distill = self.relation_distillation_loss(
                            student_feats, teacher_batch_feats.numpy()
                        )

                    # ---- SupCon Loss ----
                    loss_supcon = tf.constant(0.0)
                    if use_supcon:
                        projections = projection_head(student_feats, training=True)
                        projections_norm = tf.math.l2_normalize(projections, axis=1)
                        loss_supcon = supervised_contrastive_loss(
                            projections_norm, y_batch, temperature=supcon_temperature
                        )

                    # ---- 总损失：ArcFace + α·Distill + β·SupCon ----
                    total_loss = (loss_arcface
                                  + alpha * loss_distill
                                  + supcon_lambda * loss_supcon)

                trainable_vars = (
                    self.feature_extractor.trainable_variables +
                    self.classifier.trainable_variables
                )
                if use_supcon:
                    trainable_vars = trainable_vars + projection_head.trainable_variables

                grads = tape.gradient(total_loss, trainable_vars)
                optimizer.apply_gradients(zip(grads, trainable_vars))

            # (epoch summary omitted)

        # 范例集由 trainer 在训练后调用 select_exemplars 更新