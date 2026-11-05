# comparison/LwF.py
"""
LwF (Learning without Forgetting) 对比基线。

与本文模型的区别:
  - 无范例回放：增量训练仅使用新类数据
  - 知识蒸馏：旧模型 softmax 输出作为旧类软标签
  - 无增量 SupCon、无投影头
  - OSR：Softmax 置信度 max(softmax(logits))

与 iCaRL 的区别:
  - 无范例存储，无 NME 分类器
  - 蒸馏作用于新数据（而非新旧混合数据）
  - 使用标准 softmax 分类（而非 BCE + NME）

参考:
    Li & Hoiem, "Learning without Forgetting", ECCV 2016 / PAMI 2018.
"""

import sys
import os
import json
import random
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import DATA_PATH, RANDOM_SEED, EPOCHS_BASE, \
                     EPOCHS_INCREMENTAL, INC_LEARNING_RATE, BATCH_SIZE, \
                     INC_BATCH_SIZE, DISTILLATION_ALPHA, INC_WEIGHT_DECAY
from src.models import build_feature_extractor
from src.data_manager import LIBSDataloader
from src.metrics import analyze_feature_space
from sklearn.metrics import accuracy_score, roc_auc_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'LwF.json')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────
# 工具
# ────────────────────────────────────────────────────────────

def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)


# ────────────────────────────────────────────────────────────
# LwF 分类器
# ────────────────────────────────────────────────────────────

class LwFClassifier:
    """
    LwF 软最大分类器 + Softmax 置信度开集检测。

    分类: softmax(x · W)
    OSR:  max(softmax(logits)) → 置信度得分
    """

    def __init__(self, kernel, frozen_fe=None, old_num=None):
        self.kernel = tf.constant(kernel, dtype=tf.float32)
        self.num_classes = kernel.shape[1]
        self.old_num = old_num if old_num is not None else self.num_classes
        self.frozen_fe = frozen_fe  # 教师冻结特征提取器

    def get_logits(self, features):
        return tf.matmul(tf.cast(features, tf.float32), self.kernel)

    def predict(self, features_np):
        """返回预测标签和 softmax 置信度得分。"""
        feats = tf.constant(features_np, dtype=tf.float32)
        logits = self.get_logits(feats)
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(logits, axis=1).numpy()
        confidence = tf.reduce_max(probs, axis=1).numpy()
        return preds, confidence

    def classify_only(self, features_np):
        preds, _ = self.predict(features_np)
        return preds

    def get_old_class_logits(self, X_raw, temperature=1.0):
        """通过冻结的特征提取器 + 冻结的旧分类器计算旧类 logits。"""
        if self.frozen_fe is not None:
            feats = self.frozen_fe(X_raw, training=False)
        else:
            feats = X_raw  # fallback
        logits = tf.matmul(tf.cast(feats, tf.float32),
                          self.kernel[:, :self.old_num])
        return logits / temperature


# ────────────────────────────────────────────────────────────
# 分类头构建与扩展
# ────────────────────────────────────────────────────────────

def _build_classifier(input_dim, num_classes):
    """构建可扩展的线性分类器（无 bias, 无 activation）。"""
    dense = tf.keras.layers.Dense(num_classes, activation=None, use_bias=False,
                                  name='classifier')
    dense.build((None, input_dim))
    return dense


def _add_new_classes(old_dense, n_new):
    """扩展分类器输出节点，旧权重保留，新权重小值随机初始化。"""
    old_kernel = old_dense.get_weights()[0]
    old_num = old_kernel.shape[1]
    new_num = old_num + n_new
    in_dim = old_kernel.shape[0]

    rng = np.random.RandomState(RANDOM_SEED)
    new_kernel = np.hstack([old_kernel,
                            rng.randn(in_dim, n_new).astype(np.float32) * 0.01])

    new_dense = tf.keras.layers.Dense(new_num, activation=None, use_bias=False,
                                      name='classifier')
    new_dense.build((None, in_dim))
    new_dense.set_weights([new_kernel])
    return new_dense


# ────────────────────────────────────────────────────────────
# 训练
# ────────────────────────────────────────────────────────────

def lwf_train_base(feature_extractor, dense, X_train_cnn, y_train_mapped,
                   epochs, lr):
    """LwF 基类训练: Softmax + CE。"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    n_samples = len(X_train_cnn)
    num_classes = dense.get_weights()[0].shape[1]

    for epoch in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED + epoch).permutation(n_samples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, BATCH_SIZE):
            batch_idx = idx[i:i + BATCH_SIZE]
            X_b = tf.constant(X_train_cnn[batch_idx])
            y_b = tf.constant(y_train_mapped[batch_idx])

            with tf.GradientTape() as tape:
                feats = feature_extractor(X_b, training=True)
                logits = dense(feats)
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y_b, logits=logits))

            tvars = feature_extractor.trainable_variables + dense.trainable_variables
            grads = tape.gradient(loss, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
            total_loss += float(loss)
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | CE Loss: {total_loss / max(n_batches, 1):.4f}")


def lwf_train_incremental(feature_extractor, dense, X_new_cnn, y_new_mapped,
                          num_old, num_total, epochs, lr, alpha,
                          old_lwf_cls=None, temperature=2.0):
    """
    LwF 增量训练: CE(新类) + 蒸馏(旧类)，仅使用新类数据。

    L_total = (1 - λ) * L_CE_new + λ * L_distill_old

    其中:
      L_CE_new: 仅在新类上的 softmax CE（ground truth）
      L_distill_old: KL 散度，教师旧类概率 vs 学生旧类概率
      λ = α / (1 + α)，通过 config.DISTILLATION_ALPHA 控制
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                         weight_decay=INC_WEIGHT_DECAY)
    n_samples = len(X_new_cnn)

    print(f"  [LwF] alpha={alpha}, T={temperature}")

    for epoch in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED + epoch).permutation(n_samples)
        total_ce = 0.0
        total_kd = 0.0
        n_batches = 0
        for i in range(0, n_samples, INC_BATCH_SIZE):
            batch_idx = idx[i:i + INC_BATCH_SIZE]
            X_b = tf.constant(X_new_cnn[batch_idx])
            y_mapped_b = tf.constant(y_new_mapped[batch_idx])

            with tf.GradientTape() as tape:
                feats = feature_extractor(X_b, training=True)
                logits_all = dense(feats)  # (batch, num_total)

                # ── CE over ALL classes ──
                # LwF 原文：softmax 在所有类别上计算，但仅新类有 ground truth
                loss_ce = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y_mapped_b, logits=logits_all))

                # ── 蒸馏 on old classes ──
                loss_kd = tf.constant(0.0)
                if old_lwf_cls is not None and alpha > 0:
                    # 教师通过冻结的 FE + 冻结的分类器计算旧类 logits
                    teacher_logits_old = old_lwf_cls.get_old_class_logits(
                        X_b, temperature=temperature)
                    teacher_probs = tf.nn.softmax(teacher_logits_old)

                    # 学生温度缩放 softmax (旧类)
                    student_logits_old = logits_all[:, :num_old] / temperature
                    # KL 散度: sum(p_teacher * log(p_teacher / p_student))
                    loss_kd = tf.reduce_mean(
                        tf.reduce_sum(
                            teacher_probs * (tf.math.log(teacher_probs + 1e-8)
                                             - tf.nn.log_softmax(student_logits_old)),
                            axis=1))

                # LwF 原文: L = L_CE + α * L_KD
                loss_total = loss_ce + alpha * loss_kd

            tvars = feature_extractor.trainable_variables + dense.trainable_variables
            grads = tape.gradient(loss_total, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
            total_ce += float(loss_ce)
            total_kd += float(loss_kd)
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | CE: {total_ce / max(n_batches, 1):.4f} "
                  f"| KD: {total_kd / max(n_batches, 1):.4f}")


# ────────────────────────────────────────────────────────────
# 评估
# ────────────────────────────────────────────────────────────

def evaluate_osr_distance(feature_extractor, lwf_cls, X_known, y_known,
                          X_unknown, label_map):
    """
    基于分类器权重距离的开集识别评估。

    得分: score(x) = -min_c (1 - cos(f(x), w_c))
    即 -min_c 余弦距离。
    """
    feats_known = feature_extractor.predict(X_known[..., np.newaxis], verbose=0)
    feats_unknown = feature_extractor.predict(X_unknown[..., np.newaxis], verbose=0)

    y_known_mapped = np.array([label_map[g] for g in y_known])

    _, scores_known = lwf_cls.predict(feats_known)
    _, scores_unknown = lwf_cls.predict(feats_unknown)

    # 分类准确率
    preds_known = lwf_cls.classify_only(feats_known)
    known_cls_acc = float(np.mean(preds_known == y_known_mapped))

    # AUROC
    all_scores = np.concatenate([scores_known, scores_unknown])
    y_true = np.concatenate([np.ones(len(scores_known)),
                              np.zeros(len(scores_unknown))])
    try:
        auroc = float(roc_auc_score(y_true, all_scores))
    except ValueError:
        auroc = None

    # ── 阈值扫参 ──
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    thresholds = np.percentile(all_scores, percentiles)
    abs_thresholds = np.linspace(all_scores.min(), all_scores.max(), 6)
    all_thresholds = sorted(set(np.concatenate([thresholds, abs_thresholds])))

    threshold_sweep = []
    n_known = len(scores_known)
    n_unknown = len(scores_unknown)

    for t in all_thresholds:
        tp = int(np.sum(scores_known >= t))
        tn = int(np.sum(scores_unknown < t))
        fn = n_known - tp
        fp = n_unknown - tn

        kr = tp / n_known if n_known > 0 else 0.0
        ur = tn / n_unknown if n_unknown > 0 else 0.0
        g_mean = np.sqrt(kr * ur)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * kr / (prec + kr) if (prec + kr) > 0 else 0.0

        threshold_sweep.append({
            'threshold': round(float(t), 4),
            'known_recall': round(kr, 4),
            'unknown_recall': round(ur, 4),
            'g_mean': round(float(g_mean), 4),
            'f1': round(f1, 4),
        })

    best_g = max(threshold_sweep, key=lambda x: x['g_mean']) if threshold_sweep else None

    return {
        'confidence_auroc': auroc,
        'known_cls_acc': known_cls_acc,
        'threshold_sweep': threshold_sweep,
        'best_g_mean': best_g['g_mean'] if best_g else None,
        'best_g_threshold': best_g['threshold'] if best_g else None,
    }


def evaluate_cil(feature_extractor, lwf_cls, X_test, y_test, label_map):
    """CIL 阶段评估（softmax 分类）。"""
    mask = np.isin(y_test, list(label_map.keys()))
    X_known = X_test[mask]
    y_known_global = y_test[mask]
    y_known_mapped = np.array([label_map[g] for g in y_known_global])

    if len(X_known) == 0:
        return {'accuracy': 0.0, 'per_class_accuracy': {}, 'feature_stats': {}}

    feats = feature_extractor.predict(X_known[..., np.newaxis], verbose=0)
    preds = lwf_cls.classify_only(feats)

    acc = float(np.mean(preds == y_known_mapped))

    per_class = {}
    for c in np.unique(y_known_mapped):
        m = y_known_mapped == c
        per_class[int(c)] = float(np.mean(preds[m] == c)) if m.sum() > 0 else 0.0

    fs = analyze_feature_space(feats, y_known_mapped, sample_limit=5000)

    return {'accuracy': acc, 'per_class_accuracy': per_class, 'feature_stats': fs}


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    set_seeds()

    print("=" * 64)
    print("  LwF 对比实验 (Learning without Forgetting)")
    print("  分类器: 标准 Softmax")
    print("  增量:   CE(新类) + KL 蒸馏(旧类)，无范例回放")
    print("  OSR:    Softmax 置信度")
    print("=" * 64)

    # ── 数据 ──
    data = LIBSDataloader(DATA_PATH, random_seed=RANDOM_SEED)
    data.apply_pca(variance_ratio=config.PCA_VARIANCE, use_pca=True)

    input_shape = (data.X.shape[1], 1)
    feature_extractor = build_feature_extractor(input_shape)

    # ── 标签映射 ──
    label_map = {}
    num_known = 0

    # ── 基类训练 ──
    X0_train, y0_train = data.get_task_data(0, 'train')
    X0_test, y0_test = data.get_task_data(0, 'test')

    base_ids = sorted(data.base_ids)
    for gid in base_ids:
        if gid not in label_map:
            label_map[gid] = num_known
            num_known += 1

    y0_train_m = np.array([label_map[v] for v in y0_train])
    y0_test_m = np.array([label_map[v] for v in y0_test])

    X0_train_cnn = X0_train[..., np.newaxis]
    X0_test_cnn = X0_test[..., np.newaxis]

    feat_dim = feature_extractor(X0_train_cnn[:1]).shape[-1]
    dense = _build_classifier(feat_dim, num_known)

    print(f"\n基类训练: {num_known} 类 (Softmax + CE)")
    lwf_train_base(feature_extractor, dense, X0_train_cnn, y0_train_m,
                   EPOCHS_BASE, config.LEARNING_RATE)

    lwf_cls = LwFClassifier(dense.get_weights()[0], old_num=num_known)

    # ── Task 0 评估 ──
    r0 = evaluate_cil(feature_extractor, lwf_cls, X0_test, y0_test, label_map)
    X_unk0, _ = data.get_osr_unknown_data(0)
    osr0 = evaluate_osr_distance(feature_extractor, lwf_cls, X0_test, y0_test,
                                 X_unk0, label_map)

    fs0 = r0.get('feature_stats', {})
    stages = [{
        'task_id': 0,
        'stage': 'Task 0 (基类)',
        'cil': {
            'num_known_classes': num_known,
            'accuracy': round(r0['accuracy'], 4),
            'new_class_accuracy': None,
            'avg_forgetting': None,
            'silhouette_score': round(fs0.get('silhouette_score'), 4)
                if fs0.get('silhouette_score') is not None else None,
            'inter_intra_ratio': round(
                fs0.get('inter_class_dist_mean', 0) /
                max(fs0.get('intra_class_dist_mean', 1e-8), 1e-8), 4),
        },
        'osr': {
            'confidence_auroc': round(osr0['confidence_auroc'], 4)
                if osr0['confidence_auroc'] else None,
            'known_cls_acc': round(osr0['known_cls_acc'], 4),
            'best_g_mean': round(osr0['best_g_mean'], 4)
                if osr0['best_g_mean'] else None,
            'best_g_threshold': osr0['best_g_threshold'],
            'threshold_sweep': osr0['threshold_sweep'],
        }
    }]

    per_class_best = dict(r0.get('per_class_accuracy', {}))

    # ── 增量步骤 ──
    for task_id in range(1, len(data.inc_batches) + 1):
        print(f"\n{'─' * 50}")
        print(f"  增量步骤 {task_id}")

        X_new, y_new = data.get_task_data(task_id, 'train')
        X_new_cnn = X_new[..., np.newaxis]

        new_global_classes = sorted(np.unique(y_new))
        n_new = len(new_global_classes)

        # 扩展标签映射
        for gid in new_global_classes:
            if gid not in label_map:
                label_map[gid] = num_known
                num_known += 1

        y_new_mapped = np.array([label_map[v] for v in y_new])

        # 保存教师模型 (冻结旧特征提取器 + 旧分类器)
        old_fe = tf.keras.models.clone_model(feature_extractor)
        old_fe.set_weights(feature_extractor.get_weights())
        old_kernel = dense.get_weights()[0].copy()
        old_lwf_cls = LwFClassifier(old_kernel, frozen_fe=old_fe,
                                    old_num=old_kernel.shape[1])

        # 扩展分类器
        dense = _add_new_classes(dense, n_new)
        num_total = dense.get_weights()[0].shape[1]
        print(f"  分类器: {num_total} 输出节点")

        # 增量训练 (LwF: 仅使用新数据 + 蒸馏)
        lwf_train_incremental(
            feature_extractor, dense, X_new_cnn, y_new_mapped,
            num_old=old_kernel.shape[1], num_total=num_total,
            epochs=EPOCHS_INCREMENTAL, lr=INC_LEARNING_RATE,
            alpha=DISTILLATION_ALPHA, old_lwf_cls=old_lwf_cls,
            temperature=2.0)

        # 更新分类器 (old_num 扩展为总数)
        lwf_cls = LwFClassifier(dense.get_weights()[0], old_num=num_total)

        # CIL 评估
        X_test, y_test = data.get_task_data(task_id, 'test')
        r = evaluate_cil(feature_extractor, lwf_cls, X_test, y_test, label_map)

        # 新类准确率
        new_ids = [label_map[c] for c in new_global_classes]
        per_class = r.get('per_class_accuracy', {})
        new_acc = np.mean([per_class.get(nid, 0) for nid in new_ids])

        # 遗忘率
        old_ids = [v for k, v in label_map.items()
                   if k not in new_global_classes]
        forg = []
        for oid in old_ids:
            best = per_class_best.get(oid, per_class.get(oid, 0))
            cur = per_class.get(oid, 0)
            forg.append(max(0.0, best - cur))
        avg_forg = float(np.mean(forg)) if forg else 0.0

        for c, acc in per_class.items():
            per_class_best[c] = max(per_class_best.get(c, 0), acc)

        # OSR 评估
        X_unk, _ = data.get_osr_unknown_data(task_id)
        osr = evaluate_osr_distance(feature_extractor, lwf_cls, X_test, y_test,
                                    X_unk, label_map)

        fs = r.get('feature_stats', {})
        stages.append({
            'task_id': task_id,
            'stage': f'Task {task_id} (+{n_new}类)',
            'cil': {
                'num_known_classes': num_known,
                'accuracy': round(r['accuracy'], 4),
                'new_class_accuracy': round(new_acc, 4),
                'avg_forgetting': round(avg_forg, 4),
                'silhouette_score': round(fs.get('silhouette_score'), 4)
                    if fs.get('silhouette_score') is not None else None,
                'inter_intra_ratio': round(
                    fs.get('inter_class_dist_mean', 0) /
                    max(fs.get('intra_class_dist_mean', 1e-8), 1e-8), 4),
            },
            'osr': {
                'confidence_auroc': round(osr['confidence_auroc'], 4)
                    if osr['confidence_auroc'] else None,
                'known_cls_acc': round(osr['known_cls_acc'], 4),
                'best_g_mean': round(osr['best_g_mean'], 4)
                    if osr['best_g_mean'] else None,
                'best_g_threshold': osr['best_g_threshold'],
                'threshold_sweep': osr['threshold_sweep'],
            }
        })

        print(f"  CIL: {r['accuracy']:.4f}, 新类: {new_acc:.4f}, "
              f"遗忘: {avg_forg:.4f}, AUROC: {osr['confidence_auroc']}, "
              f"G-mean: {osr['best_g_mean']}")

    # ── 汇总 ──
    final_acc = stages[-1]['cil']['accuracy']
    avg_forg_total = float(np.mean([s['cil']['avg_forgetting'] for s in stages[1:]]))
    overall_avg = float(np.mean([s['cil']['accuracy'] for s in stages]))
    acc_drop = stages[0]['cil']['accuracy'] - final_acc

    results = {
        'model': 'LwF (Softmax + KD, w/o Replay)',
        'description': '软最大分类 + KL蒸馏(无范例回放)；OSR用Softmax置信度',
        'summary': {
            'final_accuracy': round(final_acc, 4),
            'avg_total_forgetting': round(avg_forg_total, 4),
            'overall_avg_accuracy': round(overall_avg, 4),
            'accuracy_drop': round(acc_drop, 4),
        },
        'stages': stages,
    }

    print(f"\n{'=' * 64}")
    print(f"  LwF 结果")
    s = results['summary']
    print(f"  最终准确率:     {s['final_accuracy']:.4f}")
    print(f"  平均遗忘率:     {s['avg_total_forgetting']:.4f}")
    print(f"  总体平均准确率: {s['overall_avg_accuracy']:.4f}")
    print(f"  准确率下降:     {s['accuracy_drop']:.4f}")
    print(f"{'=' * 64}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
