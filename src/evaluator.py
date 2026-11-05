# src/evaluator.py
"""
评估模块：提供闭集分类、开集识别、增量学习全阶段评估功能。
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from src.metrics import analyze_feature_space, compute_per_class_accuracy
from src.config import ARCFACE_S



# ═══════════════════════════════════════════════════════════════
# ConfidenceDetector：纯置信度开集检测器（论文第 3.5 节）
# ═══════════════════════════════════════════════════════════════

class ConfidenceDetector:
    """
    纯 Softmax 置信度开集检测器 —— 论文方法。

    决策规则:
        max(softmax(ArcFace_logits / T)) >= threshold → 已知类
        否则 → 未知类（拒识）

    温度 T > 1 缓解 ArcFace 高 s 值导致的过自信问题，
    使未知样本的 softmax 分布更均匀，max prob 降低。

    用法:
        detector = ConfidenceDetector(classifier, threshold=0.95)
        detector.fit()
        labels, is_unknown = detector.predict(features_norm)
    """

    def __init__(self, classifier, threshold=0.95):
        self.classifier = classifier
        self.threshold = threshold
        self.fitted = False

    def fit(self, X=None, y=None):
        self.fitted = True
        return self

    def predict(self, features_norm):
        logits = self.classifier.arcface_logits(features_norm, s=ARCFACE_S).numpy()
        probs = _stable_softmax(logits)
        max_conf = np.max(probs, axis=1)
        is_known = max_conf >= self.threshold
        labels = np.where(is_known, np.argmax(logits, axis=1), -1)
        return labels, ~is_known


# ═══════════════════════════════════════════════════════════════
# 纯置信度开集识别评估（论文方法）
# ═══════════════════════════════════════════════════════════════

def _stable_softmax(logits):
    """数值稳定的 softmax"""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def evaluate_osr_confidence(feature_extractor, classifier,
                            X_known, y_known, X_unknown,
                            label_map=None,
                            confidence_threshold=0.85):
    """
    纯置信度（Softmax）开集识别评估 —— 论文第 3.5 节方法。

    决策规则:
        max(softmax(ArcFace_logits)) >= confidence_threshold → 已知类
        否则 → 未知类（拒识）

    不依赖距离检测器、类中心或任何距离法组件。

    参数:
        feature_extractor: Keras Model
        classifier: DynamicClassifier
        X_known, y_known: 已知类测试数据
        X_unknown: 未知类测试数据
        label_map: {global_label: mapped_label}
        confidence_threshold: Softmax 置信度阈值，默认 0.85

    返回:
        dict: 与 evaluate_osr_comprehensive 结构一致
    """
    # ---- 提取特征 ----
    feats_known = feature_extractor.predict(X_known[..., np.newaxis], verbose=0)
    feats_unknown = feature_extractor.predict(X_unknown[..., np.newaxis], verbose=0)

    # ---- 标签映射 ----
    if label_map is not None:
        y_known_mapped = np.array([label_map[g] for g in y_known])
        reverse_map = {v: k for k, v in label_map.items()}
    else:
        y_known_mapped = y_known
        reverse_map = {l: l for l in np.unique(y_known_mapped)}

    # ---- ArcFace 推理 logits ----
    feats_known_norm = tf.math.l2_normalize(feats_known, axis=1)
    feats_unknown_norm = tf.math.l2_normalize(feats_unknown, axis=1)
    logits_known = classifier.arcface_logits(feats_known_norm, s=ARCFACE_S).numpy()
    logits_unknown = classifier.arcface_logits(feats_unknown_norm, s=ARCFACE_S).numpy()

    # ---- Softmax 置信度 ----
    probs_known = _stable_softmax(logits_known)
    probs_unknown = _stable_softmax(logits_unknown)
    conf_known = np.max(probs_known, axis=1)
    conf_unknown = np.max(probs_unknown, axis=1)

    known_accepted = conf_known >= confidence_threshold
    known_rejected = ~known_accepted
    unknown_accepted = conf_unknown >= confidence_threshold
    unknown_rejected = ~unknown_accepted

    preds_known = np.argmax(logits_known, axis=1)

    n_known = len(conf_known)
    n_unknown = len(conf_unknown)

    # ═══════════════════════════════════════════════════════════════
    # 阶段 1：已知类测试集独立评估
    # ═══════════════════════════════════════════════════════════════
    tp = int(np.sum(known_accepted))
    fn = int(np.sum(known_rejected))
    known_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    known_cls_acc = float(np.mean(
        preds_known[known_accepted] == y_known_mapped[known_accepted]
    )) if tp > 0 else 0.0

    known_eval = {
        'n_total': n_known,
        'n_accepted': tp,
        'n_rejected': fn,
        'known_recall': float(known_recall),
        'known_cls_acc': known_cls_acc,
    }

    # ═══════════════════════════════════════════════════════════════
    # 阶段 2：未知类数据集独立评估
    # ═══════════════════════════════════════════════════════════════
    tn = int(np.sum(unknown_rejected))
    fp = int(np.sum(unknown_accepted))
    unknown_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    unknown_eval = {
        'n_total': n_unknown,
        'n_rejected': tn,
        'n_accepted': fp,
        'unknown_recall': float(unknown_recall),
    }

    # ═══════════════════════════════════════════════════════════════
    # 阶段 3：混合数据集综合评估
    # ═══════════════════════════════════════════════════════════════
    n_total = n_known + n_unknown
    overall_acc = (tp + tn) / n_total if n_total > 0 else 0.0
    known_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * known_precision * known_recall /
          (known_precision + known_recall)) if (known_precision + known_recall) > 0 else 0.0

    mixed_eval = {
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'n_known': n_known, 'n_unknown': n_unknown, 'n_total': n_total,
        'overall_acc': float(overall_acc),
        'known_recall': float(known_recall),
        'known_precision': float(known_precision),
        'unknown_recall': float(unknown_recall),
        'known_cls_acc': known_cls_acc,
        'f1': float(f1),
    }

    # ═══════════════════════════════════════════════════════════════
    # 阶段 4：置信度 AUROC
    # ═══════════════════════════════════════════════════════════════
    all_conf = np.concatenate([conf_known, conf_unknown])
    y_true = np.concatenate([np.ones(n_known), np.zeros(n_unknown)])
    try:
        confidence_auroc = float(roc_auc_score(y_true, all_conf))
    except ValueError:
        confidence_auroc = None

    auroc = {'distance_auroc': None, 'confidence_auroc': confidence_auroc}

    # ═══════════════════════════════════════════════════════════════
    # 阶段 5：置信度阈值扫参
    # ═══════════════════════════════════════════════════════════════
    all_logits = np.concatenate([logits_known, logits_unknown], axis=0)
    all_probs_sweep = _stable_softmax(all_logits)
    all_conf_sweep = np.max(all_probs_sweep, axis=1)

    threshold_sweep = []
    for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]:
        pred_known = all_conf_sweep >= t
        tp_s = int(np.sum(pred_known & (y_true == 1)))
        fn_s = int(np.sum(~pred_known & (y_true == 1)))
        fp_s = int(np.sum(pred_known & (y_true == 0)))
        tn_s = int(np.sum(~pred_known & (y_true == 0)))
        kr = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        ur = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0
        prec = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        f1_s = 2 * prec * kr / (prec + kr) if (prec + kr) > 0 else 0.0
        threshold_sweep.append({
            'threshold': t,
            'known_recall': round(kr, 4),
            'unknown_recall': round(ur, 4),
            'f1': round(f1_s, 4),
        })

    # ═══════════════════════════════════════════════════════════════
    # 每类已知准确率（含拒识惩罚）
    # ═══════════════════════════════════════════════════════════════
    per_class_acc = {}
    for c in np.unique(y_known_mapped):
        mask_c = (y_known_mapped == c)
        n_c = int(np.sum(mask_c))
        correct_c = int(np.sum(mask_c & known_accepted & (preds_known == y_known_mapped)))
        global_label = reverse_map.get(c, c)
        per_class_acc[global_label] = float(correct_c / n_c) if n_c > 0 else 0.0

    return {
        'known_eval': known_eval,
        'unknown_eval': unknown_eval,
        'mixed_eval': mixed_eval,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'known_recall': float(known_recall),
        'known_precision': float(known_precision),
        'known_cls_acc': known_cls_acc,
        'unknown_recall': float(unknown_recall),
        'f1': float(f1),
        'overall_acc': float(overall_acc),
        'n_known_total': n_known,
        'n_unknown_total': n_unknown,
        'per_class_known_accuracy': per_class_acc,
        'auroc': auroc,
        'threshold_sweep': threshold_sweep,
    }




# ═══════════════════════════════════════════════════════════════
# 增量学习全阶段评估
# ═══════════════════════════════════════════════════════════════

def evaluate_full_stage(feature_extractor, classifier, X_test, y_test,
                        label_map, sample_limit=5000):
    """
    对一个阶段（所有已知类）做完整评估。

    参数：
        feature_extractor: Keras Model, 特征提取器
        classifier: DynamicClassifier, 分类器
        X_test: np.ndarray, 测试数据 (n, dim)，不含通道维
        y_test: np.ndarray, 测试标签（全局标签）
        label_map: dict, {global_label: mapped_label}
        sample_limit: int, 特征空间分析的最大采样数

    返回：
        dict: {
            'accuracy': float,
            'confusion_matrix': np.ndarray,
            'per_class_accuracy': dict,
            'feature_stats': dict,
            'predictions': np.ndarray,
        }
    """
    # 标签映射
    known_global_ids = list(label_map.keys())
    mask = np.isin(y_test, known_global_ids)
    X_known = X_test[mask]
    y_known_global = y_test[mask]
    y_known_mapped = np.array([label_map[g] for g in y_known_global])

    if len(X_known) == 0:
        print("[警告] 没有已知类测试样本！")
        return {
            'accuracy': 0.0,
            'confusion_matrix': np.zeros((len(label_map), len(label_map))),
            'per_class_accuracy': {},
            'feature_stats': {},
            'predictions': np.array([]),
        }

    # 前向传播（ArcFace 归一化推理）
    X_cnn = X_known[..., np.newaxis]
    features = feature_extractor.predict(X_cnn, verbose=0)
    features_norm = tf.math.l2_normalize(features, axis=1)
    logits = classifier.arcface_logits(features_norm, s=ARCFACE_S)
    preds_mapped = np.argmax(logits, axis=1)

    # 整体准确率
    acc = accuracy_score(y_known_mapped, preds_mapped)

    # 混淆矩阵
    num_classes = len(label_map)
    cm = confusion_matrix(y_known_mapped, preds_mapped,
                          labels=list(range(num_classes)))

    # 每类准确率
    per_class = compute_per_class_accuracy(y_known_mapped, preds_mapped,
                                           class_ids=list(range(num_classes)))

    # 特征空间结构
    feature_stats = analyze_feature_space(features, y_known_mapped,
                                          sample_limit=sample_limit)

    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class,
        'feature_stats': feature_stats,
        'predictions': preds_mapped,
    }


def evaluate_new_classes(feature_extractor, classifier, X_test, y_test,
                         label_map, new_global_ids):
    """
    仅评估新增类别的准确率。

    参数：
        new_global_ids: list, 新类的全局标签

    返回：
        float, 新类的平均准确率
    """
    new_mapped_ids = [label_map[g] for g in new_global_ids if g in label_map]
    if not new_mapped_ids:
        return 0.0

    mask = np.isin(y_test, new_global_ids)
    if mask.sum() == 0:
        return 0.0

    X_new = X_test[mask][..., np.newaxis]
    y_new_mapped = np.array([label_map[g] for g in y_test[mask]])

    features = feature_extractor.predict(X_new, verbose=0)
    features_norm = tf.math.l2_normalize(features, axis=1)
    logits = classifier.arcface_logits(features_norm, s=ARCFACE_S)
    preds = np.argmax(logits, axis=1)

    return accuracy_score(y_new_mapped, preds)


def evaluate_old_classes(feature_extractor, classifier, X_test, y_test,
                         label_map, old_global_ids):
    """
    仅评估旧类别的准确率，并返回每类准确率（用于计算遗忘率）。

    返回：
        (avg_accuracy, per_class_accuracy_dict)
    """
    old_mapped_ids = [label_map[g] for g in old_global_ids if g in label_map]
    if not old_mapped_ids:
        return 0.0, {}

    mask = np.isin(y_test, old_global_ids)
    if mask.sum() == 0:
        return 0.0, {g: 0.0 for g in old_global_ids}

    X_old = X_test[mask][..., np.newaxis]
    y_old_mapped = np.array([label_map[g] for g in y_test[mask]])

    features = feature_extractor.predict(X_old, verbose=0)
    features_norm = tf.math.l2_normalize(features, axis=1)
    logits = classifier.arcface_logits(features_norm, s=ARCFACE_S)
    preds = np.argmax(logits, axis=1)

    avg_acc = accuracy_score(y_old_mapped, preds)
    per_class = compute_per_class_accuracy(y_old_mapped, preds,
                                           class_ids=old_mapped_ids)

    # 将 mapped key 转回 global key
    reverse_map = {v: k for k, v in label_map.items()}
    per_class_global = {reverse_map.get(m, m): a for m, a in per_class.items()}

    return avg_acc, per_class_global


def get_class_partition(task_id, inc_batches, base_ids):
    """
    获取指定任务的「新类」和「旧类」划分。

    参数：
        task_id: int, 0=基类, 1/2/3=增量批次
        inc_batches: list of list, 增量批次的类别列表
        base_ids: list, 基类类别列表

    返回：
        (new_ids, old_ids): 两个 list
    """
    if task_id == 0:
        return base_ids, []

    new_ids = []
    old_ids = list(base_ids)
    for i in range(task_id):
        batch = inc_batches[i]
        if i == task_id - 1:
            new_ids = batch
        else:
            old_ids.extend(batch)

    return new_ids, old_ids
