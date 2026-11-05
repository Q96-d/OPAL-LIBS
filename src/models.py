# src/models.py

import tensorflow as tf
import numpy as np
from src.config import RANDOM_SEED

def build_feature_extractor(input_shape):
    """1D-CNN 特征提取器"""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(512, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(512, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return tf.keras.Model(inputs, x, name='feature_extractor')


class DynamicClassifier:
    """
    可动态扩展输出节点的分类器（纯 Python 封装，非 Keras Layer）。

    设计要点：
    - 不继承 Layer/Model，避免 Keras 跟踪系统锁定子层
    - 内部持有 Dense 层，add_classes() 时直接替换为新 Dense
    - 权重复制：旧类别保留，新类别小值随机初始化

    ArcFace 集成：
    - get_kernel() 返回原始权重矩阵 (input_dim, num_classes)，供 arcface_loss 使用
    - arcface_logits() 返回归一化 + 缩放的推理 logits，与训练时量级一致
    """
    def __init__(self, input_dim, initial_num_classes):
        self.dense = tf.keras.layers.Dense(
            initial_num_classes, activation=None, use_bias=False
        )
        self.num_classes = initial_num_classes
        self.dense.build((None, input_dim))

    def __call__(self, inputs, training=False):
        """标准 Dense 前向传播。"""
        return self.dense(inputs)

    def get_kernel(self):
        """
        返回分类器权重矩阵（tf.Variable，保留梯度）。

        返回:
            kernel: shape (input_dim, num_classes)，tf.Variable
        """
        return self.dense.kernel

    def arcface_logits(self, features_norm, s=30.0):
        """
        ArcFace 风格推理 logits：s * cosθ = s * (norm(features) @ norm(kernel))。

        参数:
            features_norm: L2 归一化后的特征，shape (batch, dim)
            s:             缩放因子，需与训练时 ARCFACE_S 一致

        返回:
            logits: shape (batch, num_classes)
        """
        kernel_norm = tf.math.l2_normalize(self.dense.kernel, axis=0)
        cos_theta = tf.matmul(features_norm, kernel_norm)
        return s * cos_theta

    def get_weights(self):
        return self.dense.get_weights()

    def set_weights(self, weights):
        self.dense.set_weights(weights)

    def save_weights(self, filepath):
        """保存 Dense 层权重到 .npz 文件"""
        kernel = self.dense.kernel.numpy()
        np.savez(filepath, kernel=kernel)

    def load_weights(self, filepath):
        """从 .npz 文件加载 Dense 层权重（自动适配输出节点数）"""
        data = np.load(filepath)
        kernel = data['kernel']
        self.dense.set_weights([kernel])
        self.num_classes = kernel.shape[-1]

    @property
    def trainable_variables(self):
        return self.dense.trainable_variables

    @property
    def trainable_weights(self):
        return self.dense.trainable_weights

    def add_classes(self, n_new):
        """增加 n_new 个输出节点，保留旧权重，新权重小值随机初始化"""
        old_kernel = self.dense.kernel.numpy()
        old_num = old_kernel.shape[1]
        new_num = old_num + n_new
        input_dim = old_kernel.shape[0]

        # 构建扩展后的权重矩阵（本地 RNG，确保可复现）
        rng = np.random.RandomState(RANDOM_SEED)
        new_kernel_val = np.hstack([
            old_kernel,
            rng.randn(input_dim, n_new).astype(np.float32) * 0.01
        ])

        # 直接用更大的 Dense 替换（纯 Python 对象，无 Keras 跟踪限制）
        self.dense = tf.keras.layers.Dense(
            new_num, activation=None, use_bias=False
        )
        self.dense.build((None, input_dim))
        self.dense.set_weights([new_kernel_val])
        self.num_classes = new_num
        print(f"分类器扩展: 输出节点从 {old_num} 增加到 {new_num}")