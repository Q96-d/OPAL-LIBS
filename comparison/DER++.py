# comparison/DERpp.py
"""
DER++ (Dark Experience Replay) 对比基线。

与本文模型共享: 1D-CNN backbone + DynamicClassifier + ArcFace 基类训练

DER++ 替换:
  - 回放存储: 水库采样, 存储 (样本, 标签, logits) 三元组
  - 增量损失: L = L_ArcFace(x_new) + α·L_ArcFace(x_buf) + β·MSE(z_cur, z_stored)
  - 无蒸馏、无 SupCon

参考:
    Buzzega et al., "Dark Experience for General Continual Learning:
    a Strong, Simple Baseline", NeurIPS 2020.
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
                     INC_BATCH_SIZE, INC_WEIGHT_DECAY, ARCFACE_S, ARCFACE_M, \
                     SUPCON_LAMBDA, SUPCON_TEMPERATURE, SUPCON_CLASSES_PER_BATCH, \
                     SUPCON_SAMPLES_PER_CLASS, PROJECTION_DIM, FEATURE_DIM, \
                     LEARNING_RATE
from src.models import build_feature_extractor, DynamicClassifier
from src.data_manager import LIBSDataloader
from src.metrics import analyze_feature_space
from src.losses import arcface_loss, supervised_contrastive_loss
from sklearn.metrics import accuracy_score, roc_auc_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'DERpp.json')
os.makedirs(RESULTS_DIR, exist_ok=True)

# DER++ 超参数（官方推荐值）
DER_ALPHA = 0.1     # 回放 CE 权重
DER_BETA = 0.5      # 回放 MSE (dark knowledge) 权重


def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)


# ────────────────────────────────────────────────────────────
# 水库采样回放缓冲区
# ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    DER++ 回放缓冲区: 标准水库采样, 存储 (x, y, z) 三元组。

    z: 任务结束时冻结的 logits（维度为当时的类别数 num_cls_old）。
    MSE 计算时自动截断为 min(num_cls_cur, num_cls_old) 维。
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.X = []       # list of np.ndarray (1, dim, 1)
        self.y = []       # list of int
        self.logits = []  # list of np.ndarray (num_cls_at_storage_time,)
        self.n_seen = 0
        self._rng = random.Random(RANDOM_SEED)

    def add_and_refresh(self, X_new, y_new, feature_extractor, classifier):
        """标准水库采样加入新数据，然后用当前模型刷新全部 logits。"""
        for i in range(len(X_new)):
            self.n_seen += 1
            if len(self.X) < self.max_size:
                self.X.append(X_new[i:i + 1])
                self.y.append(int(y_new[i]))
            else:
                j = self._rng.randint(0, self.n_seen - 1)
                if j < self.max_size:
                    self.X[j] = X_new[i:i + 1]
                    self.y[j] = int(y_new[i])
        self._refresh_logits(feature_extractor, classifier)

    def _refresh_logits(self, feature_extractor, classifier):
        if not self.X:
            return
        X_batch = np.concatenate(self.X, axis=0)
        feats = feature_extractor.predict(X_batch, verbose=0)
        feats_n = tf.math.l2_normalize(feats, axis=1)
        self.logits = list(classifier.arcface_logits(feats_n, s=ARCFACE_S).numpy())

    def get_all(self):
        if not self.X:
            return None, None, None
        return (np.concatenate(self.X, axis=0),
                np.array(self.y, dtype=np.int64),
                np.array(self.logits, dtype=np.float32))

    def __len__(self):
        return len(self.X)


# ────────────────────────────────────────────────────────────
# 训练（基类与 V3 完全一致）
# ────────────────────────────────────────────────────────────

def _build_projection_head(input_dim):
    proj_input = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(proj_input)
    x = tf.keras.layers.Dense(PROJECTION_DIM)(x)
    return tf.keras.Model(proj_input, x, name='projection_head')


def train_base(feature_extractor, classifier, projection_head, data, label_map, num_known):
    """基类训练: ArcFace + SupCon（与 V3 完全一致）。"""
    X_train, y_train = data.get_task_data(0, 'train')
    X_val, y_val = data.get_task_data(0, 'test')
    y_train_m = np.array([label_map[v] for v in y_train])
    y_val_m = np.array([label_map[v] for v in y_val])

    X_train_cnn = X_train[..., np.newaxis]
    X_val_cnn = X_val[..., np.newaxis]

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    batch_size_actual = SUPCON_CLASSES_PER_BATCH * SUPCON_SAMPLES_PER_CLASS
    unique_classes = np.unique(y_train_m)
    class_indices = {c: np.where(y_train_m == c)[0] for c in unique_classes}

    def balanced_generator():
        rng = np.random.RandomState(RANDOM_SEED)
        while True:
            chosen = rng.choice(unique_classes, SUPCON_CLASSES_PER_BATCH, replace=False)
            bi = []
            for c in chosen:
                bi.extend(rng.choice(class_indices[c], SUPCON_SAMPLES_PER_CLASS, replace=True))
            rng.shuffle(bi)
            yield X_train_cnn[bi], y_train_m[bi]

    train_ds = tf.data.Dataset.from_generator(
        balanced_generator,
        output_signature=(tf.TensorSpec(shape=(batch_size_actual, X_train_cnn.shape[1], 1),
                                        dtype=tf.float32),
                          tf.TensorSpec(shape=(batch_size_actual,), dtype=tf.int64))
    ).prefetch(tf.data.AUTOTUNE)

    steps = max(len(X_train) // batch_size_actual, 1)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_cnn, y_val_m)).batch(BATCH_SIZE)

    best_val_acc = 0.0
    patience = 0
    tvars = (feature_extractor.trainable_variables + classifier.trainable_variables
             + projection_head.trainable_variables)

    for epoch in range(EPOCHS_BASE):
        for Xb, yb in train_ds.take(steps):
            with tf.GradientTape() as tape:
                feats = feature_extractor(Xb, training=True)
                feats_n = tf.math.l2_normalize(feats, axis=1)
                loss_af = arcface_loss(feats_n, yb, classifier.get_kernel(),
                                       num_known, s=ARCFACE_S, m=ARCFACE_M)
                proj = projection_head(feats, training=True)
                proj_n = tf.math.l2_normalize(proj, axis=1)
                loss_sc = supervised_contrastive_loss(proj_n, yb, temperature=SUPCON_TEMPERATURE)
                loss = loss_af + SUPCON_LAMBDA * loss_sc
            grads = tape.gradient(loss, tvars)
            optimizer.apply_gradients(zip(grads, tvars))

        feats_v = feature_extractor(X_val_cnn, training=False)
        feats_vn = tf.math.l2_normalize(feats_v, axis=1)
        logits_v = classifier.arcface_logits(feats_vn, s=ARCFACE_S)
        val_acc = float(tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits_v, axis=1), y_val_m), tf.float32)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc; patience = 0
            os.makedirs('models', exist_ok=True)
            feature_extractor.save_weights('models/der_fe.weights.h5')
            classifier.save_weights('models/der_cls.npz')
        else:
            patience += 1
        if patience >= 20:
            print(f"早停 Epoch {epoch + 1}")
            break

    feature_extractor.load_weights('models/der_fe.weights.h5')
    classifier.load_weights('models/der_cls.npz')
    return best_val_acc


# ────────────────────────────────────────────────────────────
# 增量训练 (DER++)
# ────────────────────────────────────────────────────────────

def train_incremental_derpp(feature_extractor, classifier, buf, X_new_cnn, y_new_mapped,
                            epochs, lr):
    """
    DER++ 增量训练 (官方 Algorithm 1):

    for batch from new task:
        L += CE(x_new, y_new)
    for batch from buffer:
        L += α·CE(x_buf, y_buf) + β·MSE(z_cur, z_stored)
    """
    X_buf, y_buf, z_buf = buf.get_all()
    has_buf = X_buf is not None and len(X_buf) > 0

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=INC_WEIGHT_DECAY)
    num_classes = classifier.num_classes
    n_new = len(X_new_cnn)
    n_buf = len(X_buf) if has_buf else 0

    for epoch in range(epochs):
        new_idx = np.random.RandomState(RANDOM_SEED + epoch).permutation(n_new)
        buf_idx = np.random.RandomState(RANDOM_SEED + epoch + 1000).permutation(n_buf) if n_buf > 0 else []
        total_ce = total_ce_buf = total_mse = 0.0
        nb = 0

        # 确定 batch 数以较多者为准
        n_batches = max(max(1, int(np.ceil(n_new / INC_BATCH_SIZE))),
                        int(np.ceil(n_buf / INC_BATCH_SIZE)))

        for bi in range(n_batches):
            with tf.GradientTape() as tape:
                loss_total = tf.constant(0.0)
                loss_ce_val = tf.constant(0.0)
                loss_ce_buf_val = tf.constant(0.0)
                loss_mse_val = tf.constant(0.0)

                # ── 新数据 batch: CE ──
                if n_new > 0:
                    start = (bi * INC_BATCH_SIZE // 2) % max(n_new, 1)
                    end = min(start + INC_BATCH_SIZE // 2, n_new)
                    if end > start:
                        Xb_new = tf.constant(X_new_cnn[new_idx[start:end]])
                        yb_new = tf.constant(y_new_mapped[new_idx[start:end]])
                        feats_new = feature_extractor(Xb_new, training=True)
                        feats_n_new = tf.math.l2_normalize(feats_new, axis=1)
                        loss_ce_val = arcface_loss(feats_n_new, yb_new,
                                                   classifier.get_kernel(),
                                                   num_classes, s=ARCFACE_S, m=ARCFACE_M)
                        loss_total += loss_ce_val

                # ── Buffer batch: α·CE + β·MSE ──
                if n_buf > 0:
                    start = (bi * INC_BATCH_SIZE // 2) % n_buf
                    end = min(start + INC_BATCH_SIZE // 2, n_buf)
                    if end > start:
                        Xb_buf = tf.constant(X_buf[buf_idx[start:end]])
                        yb_buf = tf.constant(y_buf[buf_idx[start:end]])
                        feats_buf = feature_extractor(Xb_buf, training=True)
                        feats_n_buf = tf.math.l2_normalize(feats_buf, axis=1)

                        # α·CE on buffer
                        loss_ce_buf_val = arcface_loss(
                            feats_n_buf, yb_buf, classifier.get_kernel(),
                            num_classes, s=ARCFACE_S, m=ARCFACE_M)
                        loss_total += DER_ALPHA * loss_ce_buf_val

                        # β·MSE(z_cur, z_stored)
                        logits_cur = classifier.arcface_logits(feats_n_buf, s=ARCFACE_S)
                        old_dim = z_buf.shape[1]
                        overlap_dim = min(old_dim, num_classes)
                        zb = tf.constant(z_buf[buf_idx[start:end]][:, :overlap_dim])
                        loss_mse_val = tf.reduce_mean(
                            tf.reduce_sum((logits_cur[:, :overlap_dim] - zb) ** 2, axis=1))
                        loss_total += DER_BETA * loss_mse_val

            tvars = feature_extractor.trainable_variables + classifier.trainable_variables
            grads = tape.gradient(loss_total, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
            total_ce += float(loss_ce_val)
            total_ce_buf += float(loss_ce_buf_val)
            total_mse += float(loss_mse_val)
            nb += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | CE_new: {total_ce / max(nb, 1):.4f} "
                  f"| CE_buf: {total_ce_buf / max(nb, 1):.4f} "
                  f"| MSE: {total_mse / max(nb, 1):.6f}")


# ────────────────────────────────────────────────────────────
# 评估（与 V3 一致）
# ────────────────────────────────────────────────────────────

def evaluate_osr(feature_extractor, classifier, X_known, y_known, X_unknown, label_map):
    feats_known = feature_extractor.predict(X_known[..., np.newaxis], verbose=0)
    feats_unknown = feature_extractor.predict(X_unknown[..., np.newaxis], verbose=0)
    y_known_mapped = np.array([label_map[g] for g in y_known])

    def _conf(f):
        fn = tf.math.l2_normalize(f, axis=1)
        logits = classifier.arcface_logits(fn, s=ARCFACE_S).numpy()
        return np.max(tf.nn.softmax(logits).numpy(), axis=1), np.argmax(logits, axis=1)

    scores_k, preds_k = _conf(feats_known)
    scores_u, _ = _conf(feats_unknown)
    known_cls_acc = float(np.mean(preds_k == y_known_mapped))

    all_scores = np.concatenate([scores_k, scores_u])
    y_true = np.concatenate([np.ones(len(scores_k)), np.zeros(len(scores_u))])
    try:
        auroc = float(roc_auc_score(y_true, all_scores))
    except ValueError:
        auroc = None

    thresholds = sorted(np.unique(np.concatenate([
        np.percentile(all_scores, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]),
        np.linspace(all_scores.min(), all_scores.max(), 6)])))
    sweep = []
    nk, nu = len(scores_k), len(scores_u)
    for t in thresholds:
        tp = int(np.sum(scores_k >= t)); tn = int(np.sum(scores_u < t))
        kr = tp / nk if nk > 0 else 0.0; ur = tn / nu if nu > 0 else 0.0
        g = np.sqrt(kr * ur)
        prec = tp / (tp + (nu - tn)) if (tp + nu - tn) > 0 else 0.0
        f1 = 2 * prec * kr / (prec + kr) if (prec + kr) > 0 else 0.0
        sweep.append({'threshold': round(float(t), 4), 'known_recall': round(kr, 4),
                       'unknown_recall': round(ur, 4), 'g_mean': round(float(g), 4),
                       'f1': round(f1, 4)})
    best = max(sweep, key=lambda x: x['g_mean']) if sweep else None
    return {'confidence_auroc': auroc, 'known_cls_acc': known_cls_acc,
            'threshold_sweep': sweep,
            'best_g_mean': best['g_mean'] if best else None,
            'best_g_threshold': best['threshold'] if best else None}


def evaluate_cil(feature_extractor, classifier, X_test, y_test, label_map):
    mask = np.isin(y_test, list(label_map.keys()))
    Xk = X_test[mask]; y_global = y_test[mask]; y_mapped = np.array([label_map[g] for g in y_global])
    if len(Xk) == 0:
        return {'accuracy': 0.0, 'per_class_accuracy': {}, 'feature_stats': {}}
    feats = feature_extractor.predict(Xk[..., np.newaxis], verbose=0)
    feats_n = tf.math.l2_normalize(feats, axis=1)
    logits = classifier.arcface_logits(feats_n, s=ARCFACE_S).numpy()
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y_mapped))
    per_class = {}
    for c in np.unique(y_mapped):
        m = y_mapped == c
        per_class[int(c)] = float(np.mean(preds[m] == c)) if m.sum() > 0 else 0.0
    fs = analyze_feature_space(feats, y_mapped, sample_limit=5000)
    return {'accuracy': acc, 'per_class_accuracy': per_class, 'feature_stats': fs}


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    set_seeds()
    print("=" * 64)
    print("  DER++ 对比实验")
    print(f"  损失: CE(x_new) + α·CE(x_buf) + β·MSE(z_cur, z_old) (α={DER_ALPHA}, β={DER_BETA})")
    print(f"  回放: 水库采样 ({EXEMPLAR_SIZE}/类)")
    print("=" * 64)

    data = LIBSDataloader(DATA_PATH, random_seed=RANDOM_SEED)
    data.apply_pca(variance_ratio=config.PCA_VARIANCE, use_pca=True)
    input_shape = (data.X.shape[1], 1)
    feature_extractor = build_feature_extractor(input_shape)

    label_map = {}; num_known = 0
    X0_train, y0_train = data.get_task_data(0, 'train')
    X0_test, y0_test = data.get_task_data(0, 'test')
    base_ids = sorted(data.base_ids)
    for gid in base_ids:
        label_map[gid] = num_known; num_known += 1

    feat_dim = feature_extractor(X0_train[:1, ..., np.newaxis]).shape[-1]
    classifier = DynamicClassifier(feat_dim, num_known)
    proj_head = _build_projection_head(feat_dim)

    print(f"\n基类训练: {num_known} 类 (ArcFace + SupCon, 与 V3 一致)")
    train_base(feature_extractor, classifier, proj_head, data, label_map, num_known)

    # 初始化 DER++ 回放缓冲区 (与 V3 相同内存预算: 12/类)
    buf_size = EXEMPLAR_SIZE * num_known
    buf = ReplayBuffer(buf_size)
    y0_train_m = np.array([label_map[v] for v in y0_train])
    X0_cnn = X0_train[..., np.newaxis]
    buf.add_and_refresh(X0_cnn, y0_train_m, feature_extractor, classifier)

    # Task 0 评估
    r0 = evaluate_cil(feature_extractor, classifier, X0_test, y0_test, label_map)
    X_unk0, _ = data.get_osr_unknown_data(0)
    osr0 = evaluate_osr(feature_extractor, classifier, X0_test, y0_test, X_unk0, label_map)
    fs0 = r0.get('feature_stats', {})
    stages = [{'task_id': 0, 'stage': 'Task 0 (基类)', 'cil': {
        'num_known_classes': num_known, 'accuracy': round(r0['accuracy'], 4),
        'new_class_accuracy': None, 'avg_forgetting': None,
        'silhouette_score': round(fs0.get('silhouette_score'), 4)
            if fs0.get('silhouette_score') is not None else None,
        'inter_intra_ratio': round(fs0.get('inter_class_dist_mean', 0) /
            max(fs0.get('intra_class_dist_mean', 1e-8), 1e-8), 4)},
        'osr': {'confidence_auroc': round(osr0['confidence_auroc'], 4)
            if osr0['confidence_auroc'] else None,
            'known_cls_acc': round(osr0['known_cls_acc'], 4),
            'best_g_mean': round(osr0['best_g_mean'], 4) if osr0['best_g_mean'] else None,
            'best_g_threshold': osr0['best_g_threshold'],
            'threshold_sweep': osr0['threshold_sweep']}}]
    per_class_best = dict(r0.get('per_class_accuracy', {}))

    for task_id in range(1, len(data.inc_batches) + 1):
        print(f"\n{'─' * 50}\n  增量步骤 {task_id}")
        X_new, y_new = data.get_task_data(task_id, 'train')
        X_new_cnn = X_new[..., np.newaxis]
        new_classes = sorted(np.unique(y_new))
        n_new = len(new_classes)
        for gid in new_classes:
            if gid not in label_map:
                label_map[gid] = num_known; num_known += 1
        y_new_mapped = np.array([label_map[v] for v in y_new])

        classifier.add_classes(n_new)
        print(f"  分类器: {classifier.num_classes} 输出节点")

        # DER++ 增量训练
        train_incremental_derpp(feature_extractor, classifier, buf,
                                X_new_cnn, y_new_mapped,
                                EPOCHS_INCREMENTAL, INC_LEARNING_RATE)

        # 更新 buffer（标准水库采样 + 刷新 logits）
        buf.add_and_refresh(X_new_cnn, y_new_mapped, feature_extractor, classifier)

        X_test, y_test = data.get_task_data(task_id, 'test')
        r = evaluate_cil(feature_extractor, classifier, X_test, y_test, label_map)
        new_ids = [label_map[c] for c in new_classes]
        per_class = r.get('per_class_accuracy', {})
        new_acc = np.mean([per_class.get(nid, 0) for nid in new_ids])
        old_ids = [v for k, v in label_map.items() if k not in new_classes]
        forg = [max(0.0, per_class_best.get(oid, per_class.get(oid, 0)) - per_class.get(oid, 0))
                for oid in old_ids]
        avg_forg = float(np.mean(forg)) if forg else 0.0
        for c, acc in per_class.items():
            per_class_best[c] = max(per_class_best.get(c, 0), acc)

        X_unk, _ = data.get_osr_unknown_data(task_id)
        osr = evaluate_osr(feature_extractor, classifier, X_test, y_test, X_unk, label_map)
        fs = r.get('feature_stats', {})
        stages.append({'task_id': task_id, 'stage': f'Task {task_id} (+{n_new}类)',
            'cil': {'num_known_classes': num_known,
                'accuracy': round(r['accuracy'], 4),
                'new_class_accuracy': round(new_acc, 4),
                'avg_forgetting': round(avg_forg, 4),
                'silhouette_score': round(fs.get('silhouette_score'), 4)
                    if fs.get('silhouette_score') is not None else None,
                'inter_intra_ratio': round(fs.get('inter_class_dist_mean', 0) /
                    max(fs.get('intra_class_dist_mean', 1e-8), 1e-8), 4)},
            'osr': {'confidence_auroc': round(osr['confidence_auroc'], 4)
                if osr['confidence_auroc'] else None,
                'known_cls_acc': round(osr['known_cls_acc'], 4),
                'best_g_mean': round(osr['best_g_mean'], 4) if osr['best_g_mean'] else None,
                'best_g_threshold': osr['best_g_threshold'],
                'threshold_sweep': osr['threshold_sweep']}})
        print(f"  CIL: {r['accuracy']:.4f}, 新类: {new_acc:.4f}, "
              f"遗忘: {avg_forg:.4f}, AUROC: {osr['confidence_auroc']}, "
              f"G-mean: {osr['best_g_mean']}")

    final_acc = stages[-1]['cil']['accuracy']
    avg_forg_total = float(np.mean([s['cil']['avg_forgetting'] for s in stages[1:]]))
    overall_avg = float(np.mean([s['cil']['accuracy'] for s in stages]))
    results = {'model': 'DER++ (ArcFace + Dark Experience Replay)',
        'description': f'水库采样回放(12/类) + CE_replay(α={DER_ALPHA}) + MSE_logits(β={DER_BETA})',
        'summary': {'final_accuracy': round(final_acc, 4),
            'avg_total_forgetting': round(avg_forg_total, 4),
            'overall_avg_accuracy': round(overall_avg, 4),
            'accuracy_drop': round(stages[0]['cil']['accuracy'] - final_acc, 4)},
        'stages': stages}

    print(f"\n{'=' * 64}\n  DER++ 结果")
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
