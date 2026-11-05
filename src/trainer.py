# src/trainer.py

import os
import numpy as np
import tensorflow as tf
from src.models import build_feature_extractor, DynamicClassifier
from src.incremental import IncrementalLearner
from src.losses import supervised_contrastive_loss, arcface_loss
from src.evaluator import (
    ConfidenceDetector,
    evaluate_full_stage,
    evaluate_osr_confidence,
    get_class_partition,
)
from src.metrics import IncrementalMetrics
from src.config import *


class OSR_CIL_Trainer:
    """
    开集识别 + 类增量学习 (Open-Set Recognition & Class-Incremental Learning) 训练器。

    追踪的指标（按阶段）：
      Task 0（基类 80）: 初始准确率、混淆矩阵、特征空间结构
      Task 1（+5 → 85）: 新类学习准确率、旧类遗忘率、平均准确率
      Task 2（+5 → 90）: 新类学习准确率、旧类遗忘率、平均准确率
      Task 3（+5 → 95）: 最终准确率、总遗忘率、平均准确率
    """

    def __init__(self, data_manager):
        self.data = data_manager
        self.input_shape = (self.data.X.shape[1], 1)

        # 标签映射：全局标签 -> 局部索引
        self.label_map = {}
        self.num_known_classes = 0

        # 模型组件
        self.feature_extractor = build_feature_extractor(self.input_shape)
        self.classifier = None
        self.projection_head = None  # 对比学习投影头（仅训练时使用）
        self.incremental = None
        self.osr_detector = None     # 纯置信度开集检测器

        # 指标追踪器
        self.metrics_tracker = IncrementalMetrics(save_dir='results/')

    # ──────────────────────────────────────────────
    # 投影头（对比学习专用）
    # ──────────────────────────────────────────────
    def _build_projection_head(self, input_dim):
        """构建投影头 MLP: input_dim → 128 → PROJECTION_DIM"""
        proj_input = tf.keras.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(128, activation='relu')(proj_input)
        x = tf.keras.layers.Dense(PROJECTION_DIM)(x)
        return tf.keras.Model(proj_input, x, name='projection_head')

    def _evaluate_accuracy(self, dataset):
        """计算验证集准确率（使用 ArcFace 归一化 logits，与训练一致）"""
        total_correct = 0
        total_samples = 0
        for X_batch, y_batch in dataset:
            features = self.feature_extractor(X_batch, training=False)
            features_norm = tf.math.l2_normalize(features, axis=1)
            logits = self.classifier.arcface_logits(features_norm, s=ARCFACE_S)
            preds = tf.argmax(logits, axis=1)
            total_correct += tf.reduce_sum(
                tf.cast(tf.equal(preds, y_batch), tf.int32)
            ).numpy()
            total_samples += X_batch.shape[0]
        return total_correct / total_samples if total_samples > 0 else 0.0

    def _init_exemplars(self):
        """初始化增量学习器、范例集和开集检测器"""
        X_train, y_train = self.data.get_task_data(0, 'train')

        # 初始化增量学习器
        self.incremental = IncrementalLearner(self.feature_extractor, self.classifier)

        # 初始化开集检测器（纯置信度，论文第 3.5 节）
        self.osr_detector = ConfidenceDetector(
            self.classifier, threshold=OSR_CONFIDENCE_THRESHOLD
        )

        # 选择基类范例集（使用映射后的标签，确保与分类器输出索引一致）
        for cid in np.unique(y_train):
            idx = np.where(y_train == cid)[0]
            mapped_cid = self.label_map[cid]
            mapped_y = np.array([self.label_map[val] for val in y_train[idx]])
            self.incremental.select_exemplars(
                X_train[idx], mapped_y, mapped_cid, m=EXEMPLAR_SIZE
            )

        print("范例集与开集检测器初始化完成。")

    # ──────────────────────────────────────────────
    # 基类训练（双头架构: 分类器 + 投影头）
    # ──────────────────────────────────────────────
    def train_base(self):
        print("=" * 60)
        print("训练基类模型（双头: CE Loss + SupCon Loss）...")
        print("=" * 60)

        X_train, y_train = self.data.get_task_data(0, 'train')
        X_val, y_val = self.data.get_task_data(0, 'test')

        # ---- 标签映射 ----
        base_ids = self.data.base_ids
        self.label_map = {old: new for new, old in enumerate(base_ids)}
        self.num_known_classes = len(base_ids)

        y_train_mapped = np.array([self.label_map[val] for val in y_train])
        y_val_mapped = np.array([self.label_map[val] for val in y_val])

        assert y_train_mapped.max() < self.num_known_classes, \
            f"训练标签映射错误: max={y_train_mapped.max()}, num_classes={self.num_known_classes}"
        assert y_val_mapped.max() < self.num_known_classes, \
            f"验证标签映射错误: max={y_val_mapped.max()}, num_classes={self.num_known_classes}"
        print(f"标签映射完成: {self.num_known_classes} 个基类, 标签范围 [0, {self.num_known_classes})")

        # ---- 构建模型组件 ----
        sample_features = self.feature_extractor(X_train[:1, ..., np.newaxis])
        feat_dim = sample_features.shape[-1]

        self.classifier = DynamicClassifier(feat_dim, len(base_ids))
        self.projection_head = self._build_projection_head(feat_dim)

        # ---- 优化器和损失 ----
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        # ArcFace 超参数
        arcface_s = ARCFACE_S
        arcface_m = ARCFACE_M

        trainable_vars = (
            self.feature_extractor.trainable_variables +
            self.classifier.trainable_variables +
            self.projection_head.trainable_variables
        )
        total_params = sum(tf.size(v).numpy() for v in trainable_vars)
        print(f"可训练参数总数: {total_params}")
        print(f"  - 特征提取器: {sum(tf.size(v).numpy() for v in self.feature_extractor.trainable_variables)}")
        print(f"  - 分类器:     {sum(tf.size(v).numpy() for v in self.classifier.trainable_variables)}")
        print(f"  - 投影头:     {sum(tf.size(v).numpy() for v in self.projection_head.trainable_variables)}")
        print(f"训练损失: ArcFace (s={arcface_s}, m={arcface_m}) + λ·SupCon (λ={SUPCON_LAMBDA})")

        # ---- 数据集（平衡采样） ----
        X_train_cnn = X_train[..., np.newaxis]
        X_val_cnn = X_val[..., np.newaxis]

        batch_size_actual = SUPCON_CLASSES_PER_BATCH * SUPCON_SAMPLES_PER_CLASS
        print(f"Batch 采样策略: {SUPCON_CLASSES_PER_BATCH} 类 × "
              f"{SUPCON_SAMPLES_PER_CLASS} 样本/类 = {batch_size_actual}")

        def balanced_generator():
            rng = np.random.RandomState(RANDOM_SEED)  # 本地 RNG，隔离全局状态
            unique_classes = np.unique(y_train_mapped)
            class_indices = {c: np.where(y_train_mapped == c)[0] for c in unique_classes}
            while True:
                chosen = rng.choice(unique_classes, SUPCON_CLASSES_PER_BATCH, replace=False)
                batch_indices = []
                for c in chosen:
                    idx = rng.choice(class_indices[c], SUPCON_SAMPLES_PER_CLASS, replace=True)
                    batch_indices.extend(idx)
                rng.shuffle(batch_indices)
                yield X_train_cnn[batch_indices], y_train_mapped[batch_indices]

        train_dataset = tf.data.Dataset.from_generator(
            balanced_generator,
            output_signature=(
                tf.TensorSpec(shape=(batch_size_actual, X_train_cnn.shape[1], 1), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size_actual,), dtype=tf.int64),
            )
        ).prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = max(len(X_train) // batch_size_actual, 1)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val_cnn, y_val_mapped)).batch(BATCH_SIZE)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(EPOCHS_BASE):
            epoch_loss_arcface = 0.0
            epoch_loss_supcon = 0.0
            epoch_loss_total = 0.0
            num_batches = 0
            first_batch = (epoch == 0)

            for X_batch, y_batch in train_dataset.take(steps_per_epoch):
                if first_batch and num_batches == 0:
                    y_np = y_batch.numpy()
                    unique, counts = np.unique(y_np, return_counts=True)
                    print(f"[Batch 0] 类别数={len(unique)}, "
                          f"每类样本数: {counts[0]} (期望={SUPCON_SAMPLES_PER_CLASS})")

                with tf.GradientTape() as tape:
                    # 特征提取
                    features = self.feature_extractor(X_batch, training=True)

                    # ── ArcFace 损失（替代 CE）──
                    # L2 归一化特征，使点积等于 cosθ
                    features_norm = tf.math.l2_normalize(features, axis=1)
                    loss_arcface = arcface_loss(
                        features_norm, y_batch,
                        self.classifier.get_kernel(),
                        self.num_known_classes,
                        s=arcface_s, m=arcface_m
                    )

                    # ── SupCon 损失（保持不变）──
                    projections = self.projection_head(features, training=True)
                    projections_norm = tf.math.l2_normalize(projections, axis=1)
                    loss_supcon = supervised_contrastive_loss(
                        projections_norm, y_batch, temperature=SUPCON_TEMPERATURE
                    )

                    # ── 总损失：ArcFace + λ·SupCon ──
                    loss_total = loss_arcface + SUPCON_LAMBDA * loss_supcon

                trainable_vars = (
                    self.feature_extractor.trainable_variables +
                    self.classifier.trainable_variables +
                    self.projection_head.trainable_variables
                )
                grads = tape.gradient(loss_total, trainable_vars)
                optimizer.apply_gradients(zip(grads, trainable_vars))

                epoch_loss_arcface += loss_arcface.numpy()
                epoch_loss_supcon += loss_supcon.numpy()
                epoch_loss_total += loss_total.numpy()
                num_batches += 1

            val_acc = self._evaluate_accuracy(val_dataset)

            avg_loss_arcface = epoch_loss_arcface / max(num_batches, 1)
            avg_loss_supcon = epoch_loss_supcon / max(num_batches, 1)
            avg_loss_total = epoch_loss_total / max(num_batches, 1)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS_BASE} | "
                      f"ArcFace: {avg_loss_arcface:.4f} | SupCon: {avg_loss_supcon:.4f} | "
                      f"Total: {avg_loss_total:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('models', exist_ok=True)
                self.feature_extractor.save_weights('models/feature_extractor_best.weights.h5')
                self.classifier.save_weights('models/classifier_best.npz')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 20:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break

        # ---- 加载最佳权重 ----
        self.feature_extractor.load_weights('models/feature_extractor_best.weights.h5')
        self.classifier.load_weights('models/classifier_best.npz')

        # ---- 保存完整模型 ----
        inputs = self.feature_extractor.input
        features = self.feature_extractor(inputs)
        outputs = self.classifier.dense(features)
        full_model = tf.keras.Model(inputs, outputs)
        os.makedirs('models', exist_ok=True)
        full_model.save('models/base_model.keras')
        print("基类模型已保存至 models/base_model.keras")

        # ---- 初始化范例集 ----
        self._init_exemplars()

        # ---- Task 0 完整评估 ----
        self._evaluate_and_record_task0()

    # ──────────────────────────────────────────────
    # 增量步骤
    # ──────────────────────────────────────────────
    def incremental_step(self, task_id):
        print(f"\n" + "=" * 60)
        print(f"增量步骤 {task_id}...")
        print("=" * 60)

        X_new, y_new = self.data.get_task_data(task_id, 'train')
        X_new_cnn = X_new[..., np.newaxis]

        # 1. 为新类分配局部索引
        new_classes = np.unique(y_new)
        for old_label in new_classes:
            if old_label not in self.label_map:
                self.label_map[old_label] = self.num_known_classes
                self.num_known_classes += 1

        y_new_mapped = np.array([self.label_map[val] for val in y_new])

        # ---- 标签映射诊断 ----
        print(f"  新增类别全局标签: {new_classes.tolist()}")
        print(f"  映射后局部索引:   {[self.label_map[c] for c in new_classes]}")
        print(f"  当前标签映射总数: {len(self.label_map)}")
        print(f"  新增前分类器输出节点数: {self.classifier.num_classes}")

        # 2. 扩展分类器
        self.classifier.add_classes(len(new_classes))

        print(f"  扩展后分类器输出节点数: {self.classifier.num_classes}")
        assert self.classifier.num_classes == self.num_known_classes, \
            f"分类器节点数 ({self.classifier.num_classes}) 与已知类别数 ({self.num_known_classes}) 不一致！"

        # 3. 增量训练 (含 SupCon)
        self.incremental.incremental_train(
            X_new_cnn, y_new_mapped,
            epochs=EPOCHS_INCREMENTAL,
            lr=INC_LEARNING_RATE,
            alpha=DISTILLATION_ALPHA,
            projection_head=self.projection_head,
            supcon_lambda=INC_SUPCON_LAMBDA,
            supcon_temperature=SUPCON_TEMPERATURE
        )

        # 4. 为新类选择范例
        for cid in new_classes:
            idx = np.where(y_new == cid)[0]
            mapped_cid = self.label_map[cid]
            mapped_y = np.array([self.label_map[val] for val in y_new[idx]])
            self.incremental.select_exemplars(X_new[idx], mapped_y, mapped_cid, m=EXEMPLAR_SIZE)

        # 5. 增量阶段完整评估
        self._evaluate_and_record_incremental(task_id)

        # 6. 开集识别评估（纯置信度）
        self._evaluate_osr(task_id)

    # ──────────────────────────────────────────────
    # 评估与记录（指标追踪系统）
    # ──────────────────────────────────────────────

    def _evaluate_and_record_task0(self):
        """
        Task 0（基类）完整评估 → 记录到 metrics_tracker。

        指标：初始准确率、混淆矩阵、特征空间结构
        """
        print("\n" + "-" * 50)
        print("  Task 0 完整评估...")

        X_test, y_test = self.data.get_task_data(0, 'test')

        result = evaluate_full_stage(
            self.feature_extractor, self.classifier,
            X_test, y_test, self.label_map
        )

        self.metrics_tracker.record_task0(
            accuracy=result['accuracy'],
            confusion_mat=result['confusion_matrix'],
            per_class_acc=result['per_class_accuracy'],
            feature_stats=result['feature_stats'],
        )

        # ---- 开集识别评估（纯置信度） ----
        self._evaluate_osr(0)

    def _evaluate_and_record_incremental(self, task_id):
        """
        Task N（增量）完整评估 → 记录到 metrics_tracker。

        指标：新类学习准确率、旧类遗忘率、平均准确率、混淆矩阵、特征空间结构
        """
        print(f"\n  Task {task_id} 完整评估...")

        # 获取当前所有已知类的测试数据
        X_test, y_test = self.data.get_task_data(task_id, 'test')

        # 全阶段评估
        result = evaluate_full_stage(
            self.feature_extractor, self.classifier,
            X_test, y_test, self.label_map
        )

        # 划分新/旧类
        new_ids, old_ids = get_class_partition(
            task_id, self.data.inc_batches, self.data.base_ids
        )

        self.metrics_tracker.record_incremental(
            task_id=task_id,
            new_class_ids=new_ids,
            old_class_ids=old_ids,
            per_class_acc=result['per_class_accuracy'],
            confusion_mat=result['confusion_matrix'],
            feature_stats=result['feature_stats'],
        )

    # ──────────────────────────────────────────────
    # 开集识别评估（纯置信度 + 可选消融对比）
    # ──────────────────────────────────────────────
    def _evaluate_osr(self, task_id):
        """
        纯置信度（Softmax）开集识别评估。

        决策规则: max(softmax(ArcFace_logits)) >= threshold → 已知类。

        当 ENABLE_ABLATION=True 时，同时计算距离法 AUROC 用于消融对比。
        """
        X_unknown, _ = self.data.get_osr_unknown_data(task_id)
        X_known, y_known = self.data.get_task_data(task_id, 'test')

        # ── 主框架：置信度法 ──
        osr_result = evaluate_osr_confidence(
            self.feature_extractor, self.classifier,
            X_known, y_known, X_unknown,
            label_map=self.label_map,
            confidence_threshold=OSR_CONFIDENCE_THRESHOLD,
        )

        self.metrics_tracker.record_osr(task_id, osr_result)

    def final_summary(self):
        """打印并保存全阶段增量学习指标摘要"""
        return self.metrics_tracker.summary()
