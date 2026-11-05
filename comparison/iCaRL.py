# comparison/iCaRL.py
"""
iCaRL (Incremental Classifier and Representation Learning) 对比基线。

与本文模型的区别:
  - 分类方式：最近类均值 (NME) 替代可学习线性分类器 (ArcFace)
  - 训练损失：BCE + 蒸馏，替代 ArcFace + SupCon
  - 无投影头、无对比学习
  - OSR：基于距离的置信度得分

    score(x) = -min_c ||f(x) - mu_c||^2
    其中 mu_c 为类别 c 的范例特征均值（原型）。

    得分越高 → 离最近原型越近 → 越可能属于已知类。
    基于该得分计算 AUROC 以评估已知/未知区分能力。

参考:
    Rebuffi et al., "iCaRL: Incremental Classifier and Representation Learning",
    CVPR 2017.
"""

import sys
import os
import json
import random
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import DATA_PATH, RANDOM_SEED, EXEMPLAR_SIZE, EPOCHS_BASE, \
                     EPOCHS_INCREMENTAL, INC_LEARNING_RATE, BATCH_SIZE, \
                     INC_BATCH_SIZE, DISTILLATION_ALPHA, INC_WEIGHT_DECAY
from src.models import build_feature_extractor
from src.data_manager import LIBSDataloader
from src.metrics import analyze_feature_space
from sklearn.metrics import accuracy_score, roc_auc_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'iCaRL.json')
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


def _onehot(labels, depth):
    return tf.one_hot(labels, depth)


# ────────────────────────────────────────────────────────────
# iCaRL 范例管理器
# ────────────────────────────────────────────────────────────

class ExemplarManager:
    """
    iCaRL 范例管理：herding 选择 + 类原型维护。

    Herding 策略：贪心选择使 running mean 不断逼近类中心的样本，
    使得 m 个范例的均值最大程度代表整个类别的特征分布。
    """

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.exemplars = {}       # {mapped_class_id: np.ndarray (n, dim, 1)}
        self.prototypes = {}      # {mapped_class_id: np.ndarray (dim,)}

    def select_exemplars(self, X, class_id, m=EXEMPLAR_SIZE):
        """
        Herding 范例选择。

        参数:
            X: (n, dim) 或 (n, dim, 1)
            class_id: 映射后的类别 ID
            m: 每类保留的范例数
        """
        if X.ndim == 2:
            X = X[..., np.newaxis]
        features = self.feature_extractor.predict(X, verbose=0)  # (n, feat_dim)
        class_mean = features.mean(axis=0)

        selected = []
        selected_sum = np.zeros_like(class_mean)
        n_selected = min(m, len(features))

        for _ in range(n_selected):
            best_idx = -1
            best_err = float('inf')
            for i in range(len(features)):
                if i in selected:
                    continue
                temp_mean = (selected_sum + features[i]) / (len(selected) + 1)
                err = np.sum((temp_mean - class_mean) ** 2)
                if err < best_err:
                    best_err = err
                    best_idx = i
            selected.append(best_idx)
            selected_sum += features[best_idx]

        self.exemplars[class_id] = X[selected]
        self._update_prototype(class_id)

    def _update_prototype(self, class_id):
        X = self.exemplars[class_id]
        if X.ndim == 2:
            X = X[..., np.newaxis]
        features = self.feature_extractor.predict(X, verbose=0)
        self.prototypes[class_id] = features.mean(axis=0)

    def update_all_prototypes(self):
        """全量重建原型（增量训练后特征提取器参数变化时调用）。"""
        for cid in list(self.exemplars.keys()):
            self._update_prototype(cid)

    def get_exemplar_data(self):
        """返回合并的范例数据 (X, y)，用于增量训练混合。"""
        if not self.exemplars:
            return None, None
        X_list, y_list = [], []
        for cid, X_ex in self.exemplars.items():
            X_list.append(X_ex)
            y_list.append(np.full(len(X_ex), cid, dtype=np.int64))
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    def get_prototype_matrix(self):
        """返回原型矩阵 (n_classes, feat_dim) 及对应类别 ID 列表。"""
        sorted_ids = sorted(self.prototypes.keys())
        if not sorted_ids:
            return np.array([]), []
        return np.stack([self.prototypes[cid] for cid in sorted_ids]), sorted_ids


# ────────────────────────────────────────────────────────────
# iCaRL NME 分类器 & 距离置信度 OSR
# ────────────────────────────────────────────────────────────

class iCaRLClassifier:
    """
    最近类均值分类器 + 距离置信度开集检测。

    分类: argmax_c cos(f(x), mu_c)
    OSR:  score(x) = -min_c ||f(x) - mu_c||^2
    """

    def __init__(self, em: ExemplarManager):
        self.em = em

    def predict(self, features):
        proto_mat, class_ids = self.em.get_prototype_matrix()
        if len(proto_mat) == 0:
            return np.full(len(features), -1), np.zeros(len(features))

        dists = np.sum(features[:, None, :] ** 2, axis=2) \
              + np.sum(proto_mat[None, :, :] ** 2, axis=2) \
              - 2 * features @ proto_mat.T
        dists = np.maximum(dists, 0.0)

        best_idx = np.argmin(dists, axis=1)
        preds = np.array([class_ids[i] for i in best_idx])
        min_dists = np.min(dists, axis=1)
        confidence = -min_dists  # 负距离 = 置信度
        return preds, confidence

    @staticmethod
    def _softmax(x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def classify_only(self, features):
        preds, _ = self.predict(features)
        return preds


# ────────────────────────────────────────────────────────────
# 训练
# ────────────────────────────────────────────────────────────

def _build_base_head(input_dim, num_classes):
    """基类训练用 softmax 分类头（iCaRL 原文：标准 CE 训练）。"""
    head = tf.keras.layers.Dense(num_classes, activation=None,
                                 use_bias=True, name='base_head')
    head.build((None, input_dim))
    return head


def _softmax_to_sigmoid_head(softmax_head):
    """
    将 softmax 头权重复制为 sigmoid 头，用于增量训练。

    iCaRL: 基类用 softmax+CE，增量阶段转为 sigmoid+BCE+蒸馏。
    权重矩阵直接复用，仅 activation 变化。
    """
    w, b = softmax_head.get_weights()
    in_dim, num = w.shape
    sig_head = tf.keras.layers.Dense(num, activation='sigmoid',
                                     use_bias=True, name='cls_head')
    sig_head.build((None, in_dim))
    sig_head.set_weights([w, b])
    return sig_head


def _expand_head(old_head, n_new):
    """扩展 sigmoid 分类头：旧权重保留，新权重小值随机初始化。"""
    old_w, old_b = old_head.get_weights()
    old_num = old_w.shape[1]
    new_num = old_num + n_new
    in_dim = old_w.shape[0]

    rng = np.random.RandomState(RANDOM_SEED)
    new_w = np.hstack([old_w, rng.randn(in_dim, n_new).astype(np.float32) * 0.01])
    new_b = np.concatenate([old_b, np.zeros(n_new, dtype=np.float32)])

    new_head = tf.keras.layers.Dense(new_num, activation='sigmoid',
                                     use_bias=True, name='cls_head')
    new_head.build((None, in_dim))
    new_head.set_weights([new_w, new_b])
    return new_head


def icarl_train_base(feature_extractor, head, X_train_cnn, y_train_mapped,
                     epochs, lr):
    """iCaRL 基类训练：Softmax + Cross-Entropy（原文第 3.1 节）。"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    n_samples = len(X_train_cnn)

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
                logits = head(feats)  # 无 activation，由 softmax CE 内部处理
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y_b, logits=logits))

            tvars = feature_extractor.trainable_variables + head.trainable_variables
            grads = tape.gradient(loss, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
            total_loss += float(loss)
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | CE Loss: {total_loss / max(n_batches, 1):.4f}")


def icarl_train_incremental(feature_extractor, head, em, X_new_cnn, y_new_mapped,
                            num_classes, epochs, lr, alpha, teacher_head=None):
    """增量训练：BCE + 蒸馏。"""
    # 混合新旧数据
    X_list = [X_new_cnn]
    y_list = [y_new_mapped]
    X_ex, y_ex = em.get_exemplar_data()
    if X_ex is not None:
        X_list.append(X_ex)
        y_list.append(y_ex)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                         weight_decay=INC_WEIGHT_DECAY)
    n_samples = len(X_all)

    for epoch in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED + epoch).permutation(n_samples)
        total_cls = 0.0
        total_distill = 0.0
        n_batches = 0
        for i in range(0, n_samples, INC_BATCH_SIZE):
            batch_idx = idx[i:i + INC_BATCH_SIZE]
            X_b = tf.constant(X_all[batch_idx])
            y_b = tf.constant(y_all[batch_idx])

            with tf.GradientTape() as tape:
                feats = feature_extractor(X_b, training=True)
                logits = head(feats)

                loss_cls = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(_onehot(y_b, num_classes), logits))

                loss_distill = tf.constant(0.0)
                if teacher_head is not None and alpha > 0:
                    old_num = teacher_head.get_weights()[0].shape[1]
                    # 教师 sigmoid 输出作为软目标（iCaRL 标准做法）
                    teacher_probs = tf.nn.sigmoid(teacher_head(feats, training=False))
                    # 学生仅在旧类别维度上匹配教师
                    student_probs_old = tf.nn.sigmoid(logits[:, :old_num])
                    # 对所有样本蒸馏（iCaRL 原论文公式 4）
                    loss_distill = tf.reduce_mean(
                        tf.keras.losses.binary_crossentropy(teacher_probs, student_probs_old))

                loss_total = loss_cls + alpha * loss_distill

            tvars = feature_extractor.trainable_variables + head.trainable_variables
            grads = tape.gradient(loss_total, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
            total_cls += float(loss_cls)
            total_distill += float(loss_distill)
            n_batches += 1


# ────────────────────────────────────────────────────────────
# 评估
# ────────────────────────────────────────────────────────────

def evaluate_osr_distance(feature_extractor, icarl_cls, X_known, y_known,
                          X_unknown, label_map):
    """
    基于距离的开集识别评估（含阈值扫参）。

    得分: score(x) = -min_c ||f(x) - mu_c||^2
    得分越高 → 离原型越近 → 越可能属于已知类。

    返回 AUROC、阈值扫参 (KR, UR, G-mean, F1)。
    """
    feats_known = feature_extractor.predict(X_known[..., np.newaxis], verbose=0)
    feats_unknown = feature_extractor.predict(X_unknown[..., np.newaxis], verbose=0)

    y_known_mapped = np.array([label_map[g] for g in y_known])

    # 距离置信度
    _, scores_known = icarl_cls.predict(feats_known)
    _, scores_unknown = icarl_cls.predict(feats_unknown)

    # NME 分类准确率
    preds_known = icarl_cls.classify_only(feats_known)
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
    # 使用 all_scores 的百分位数作为阈值候选，避免对绝对距离尺度的依赖
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    thresholds = np.percentile(all_scores, percentiles)
    # 也加入几个绝对阈值作为参考
    abs_thresholds = np.linspace(all_scores.min(), all_scores.max(), 6)
    all_thresholds = sorted(set(np.concatenate([thresholds, abs_thresholds])))

    threshold_sweep = []
    n_known = len(scores_known)
    n_unknown = len(scores_unknown)

    for t in all_thresholds:
        known_accepted = scores_known >= t
        unknown_rejected = scores_unknown < t

        tp = int(np.sum(known_accepted))
        fn = n_known - tp
        tn = int(np.sum(unknown_rejected))
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

    # 找到最优 G-mean
    best_g = max(threshold_sweep, key=lambda x: x['g_mean']) if threshold_sweep else None

    return {
        'confidence_auroc': auroc,
        'known_cls_acc': known_cls_acc,
        'threshold_sweep': threshold_sweep,
        'best_g_mean': best_g['g_mean'] if best_g else None,
        'best_g_threshold': best_g['threshold'] if best_g else None,
        'scores_known': scores_known,
        'scores_unknown': scores_unknown,
    }


def evaluate_cil(feature_extractor, icarl_cls, X_test, y_test, label_map):
    """CIL 阶段评估（仅评估已知类）。"""
    mask = np.isin(y_test, list(label_map.keys()))
    X_known = X_test[mask]
    y_known_global = y_test[mask]
    y_known_mapped = np.array([label_map[g] for g in y_known_global])

    if len(X_known) == 0:
        return {'accuracy': 0.0, 'per_class_accuracy': {}, 'feature_stats': {}}

    feats = feature_extractor.predict(X_known[..., np.newaxis], verbose=0)
    preds = icarl_cls.classify_only(feats)

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
    print("  iCaRL 对比实验")
    print("  分类器: NME (最近类均值)")
    print("  损失:   BCE + 知识蒸馏")
    print("  OSR:    距离置信度 score(x) = -min_c||f(x)-mu_c||^2")
    print("=" * 64)

    # ── 数据 ──
    data = LIBSDataloader(DATA_PATH, random_seed=RANDOM_SEED)
    data.apply_pca(variance_ratio=config.PCA_VARIANCE, use_pca=True)

    input_shape = (data.X.shape[1], 1)
    feature_extractor = build_feature_extractor(input_shape)
    em = ExemplarManager(feature_extractor)

    # ── 标签映射 ──
    label_map = {}          # {global_label: mapped_label}
    reverse_map = {}        # {mapped_label: global_label}
    num_known = 0

    # ── 基类训练 ──
    X0_train, y0_train = data.get_task_data(0, 'train')
    X0_test, y0_test = data.get_task_data(0, 'test')

    base_ids = sorted(data.base_ids)
    for gid in base_ids:
        if gid not in label_map:
            label_map[gid] = num_known
            reverse_map[num_known] = gid
            num_known += 1

    y0_train_m = np.array([label_map[v] for v in y0_train])
    y0_test_m = np.array([label_map[v] for v in y0_test])

    X0_train_cnn = X0_train[..., np.newaxis]
    X0_test_cnn = X0_test[..., np.newaxis]

    feat_dim = feature_extractor(X0_train_cnn[:1]).shape[-1]

    # iCaRL: 基类使用 softmax + CE 训练
    base_head = _build_base_head(feat_dim, num_known)
    print(f"\n基类训练: {num_known} 类 (Softmax + CE)")
    icarl_train_base(feature_extractor, base_head, X0_train_cnn, y0_train_m,
                     EPOCHS_BASE, config.LEARNING_RATE)

    # iCaRL: 转为 sigmoid 头用于增量阶段（权重复用，activation 变化）
    head = _softmax_to_sigmoid_head(base_head)

    # ── 基类范例 ──
    for gid in base_ids:
        idx = np.where(y0_train == gid)[0]
        em.select_exemplars(X0_train[idx], label_map[gid], m=EXEMPLAR_SIZE)

    # 基类训练后重建原型
    em.update_all_prototypes()
    icarl_cls = iCaRLClassifier(em)

    # ── Task 0 评估 ──
    r0 = evaluate_cil(feature_extractor, icarl_cls, X0_test, y0_test, label_map)
    X_unk0, _ = data.get_osr_unknown_data(0)
    osr0 = evaluate_osr_distance(feature_extractor, icarl_cls, X0_test, y0_test,
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
    all_accuracies = [r0['accuracy']]

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
                reverse_map[num_known] = gid
                num_known += 1

        y_new_mapped = np.array([label_map[v] for v in y_new])

        # 保存教师
        # 保存增量前的 sigmoid 头作为教师
        old_num = head.get_weights()[0].shape[1]
        teacher_head = tf.keras.layers.Dense(old_num, activation='sigmoid',
                                             use_bias=True, name='teacher_head')
        teacher_head.build((None, feat_dim))
        teacher_head.set_weights(head.get_weights())

        # 扩展分类头
        head = _expand_head(head, n_new)

        print(f"  分类头: {head.get_weights()[0].shape[1]} 输出节点")
        print(f"  新类: {new_global_classes}, 范例类: {len(em.exemplars)}")

        # 增量训练
        icarl_train_incremental(feature_extractor, head, em, X_new_cnn, y_new_mapped,
                                num_known, EPOCHS_INCREMENTAL, INC_LEARNING_RATE,
                                DISTILLATION_ALPHA, teacher_head=teacher_head)

        # 为新类选择范例
        for gid in new_global_classes:
            idx = np.where(y_new == gid)[0]
            em.select_exemplars(X_new[idx], label_map[gid], m=EXEMPLAR_SIZE)

        # 特征提取器参数已变，重建所有原型（iCaRL 标准流程）
        em.update_all_prototypes()
        icarl_cls = iCaRLClassifier(em)

        # CIL 评估
        X_test, y_test = data.get_task_data(task_id, 'test')
        r = evaluate_cil(feature_extractor, icarl_cls, X_test, y_test, label_map)

        # 新类准确率
        new_ids = [label_map[c] for c in new_global_classes]
        per_class = r.get('per_class_accuracy', {})
        new_acc = np.mean([per_class.get(nid, 0) for nid in new_ids])

        # 遗忘率：旧类最佳准确率 vs 当前准确率
        old_ids = [v for k, v in label_map.items()
                   if k not in new_global_classes]
        forg = []
        for oid in old_ids:
            best = per_class_best.get(oid, per_class.get(oid, 0))
            cur = per_class.get(oid, 0)
            forg.append(max(0.0, best - cur))
        avg_forg = float(np.mean(forg)) if forg else 0.0

        # 更新最佳记录
        for c, acc in per_class.items():
            per_class_best[c] = max(per_class_best.get(c, 0), acc)

        # OSR 评估
        X_unk, _ = data.get_osr_unknown_data(task_id)
        osr = evaluate_osr_distance(feature_extractor, icarl_cls, X_test, y_test,
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

        all_accuracies.append(r['accuracy'])
        print(f"  CIL 准确率: {r['accuracy']:.4f}, 新类: {new_acc:.4f}, "
              f"遗忘: {avg_forg:.4f}, AUROC: {osr['confidence_auroc']}, "
              f"G-mean: {osr['best_g_mean']}")

    # ── 汇总 ──
    final_acc = stages[-1]['cil']['accuracy']
    avg_forg_total = float(np.mean([s['cil']['avg_forgetting'] for s in stages[1:]]))
    overall_avg = float(np.mean([s['cil']['accuracy'] for s in stages]))
    acc_drop = stages[0]['cil']['accuracy'] - final_acc

    results = {
        'model': 'iCaRL (NME + BCE + Distill)',
        'description': '最近类均值分类 + BCE损失 + 蒸馏 + 范例回放；OSR用Softmax置信度',
        'summary': {
            'final_accuracy': round(final_acc, 4),
            'avg_total_forgetting': round(avg_forg_total, 4),
            'overall_avg_accuracy': round(overall_avg, 4),
            'accuracy_drop': round(acc_drop, 4),
        },
        'stages': stages,
    }

    print(f"\n{'=' * 64}")
    print(f"  iCaRL 结果")
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
