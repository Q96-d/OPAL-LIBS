# comparison/EWC.py
"""
EWC (Elastic Weight Consolidation) 对比基线。

防遗忘策略: 参数重要性加权正则化（Fisher Information）
存储旧数据: ❌ 不需要
应用场景: 多任务/类别增量学习（无数据回放）

损失: L = L_CE_new + (λ/2) * Σ_t Σ_i F_{t,i} * (θ_i - θ_{t,i}^*)^2
其中 F_{t,i} 为任务 t 中参数 i 的 Fisher 信息（重要性），
θ_{t,i}^* 为任务 t 结束时的参数最优值。

参考:
    Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks",
    PNAS 2017.
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
                     INC_BATCH_SIZE, INC_WEIGHT_DECAY, ARCFACE_S, ARCFACE_M
from src.models import build_feature_extractor, DynamicClassifier
from src.data_manager import LIBSDataloader
from src.metrics import analyze_feature_space
from sklearn.metrics import accuracy_score, roc_auc_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'EWC.json')
os.makedirs(RESULTS_DIR, exist_ok=True)

# EWC 超参数
EWC_LAMBDA = 100.0         # 正则化强度 λ
EWC_FISHER_SAMPLES = 200   # Fisher 计算采样的样本数


def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)


# ────────────────────────────────────────────────────────────
# EWC 核心：Fisher 信息 + 参数约束
# ────────────────────────────────────────────────────────────

def compute_fisher(feature_extractor, classifier, X, y_mapped, n_samples=EWC_FISHER_SAMPLES):
    """
    计算模型参数的 Fisher 信息矩阵（对角近似）。

    F_i = (1/N) * Σ (∂L/∂θ_i)^2

    返回: fisher 值列表（顺序与 tvars 一致）
    """
    tvars = feature_extractor.trainable_variables + classifier.trainable_variables
    fisher = [tf.zeros_like(v) for v in tvars]

    n = min(n_samples, len(X))
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X), n, replace=False)
    X_sub = tf.constant(X[idx])
    y_sub = tf.constant(y_mapped[idx])

    bs = 32
    for i in range(0, n, bs):
        bi = slice(i, min(i + bs, n))
        with tf.GradientTape() as tape:
            logits = classifier(feature_extractor(X_sub[bi], training=False))
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_sub[bi], logits=logits))
        grads = tape.gradient(loss, tvars)
        for j, g in enumerate(grads):
            if g is not None:
                fisher[j] += tf.reduce_mean(g ** 2, axis=0)

    n_batches = max(1, int(np.ceil(n / bs)))
    for j in range(len(fisher)):
        fisher[j] /= n_batches

    return fisher, tvars


def ewc_penalty(tvars, old_params_list, fisher_list):
    """
    计算 EWC 正则化损失。

    处理分类器扩展：若 v 的形状变大，仅对重叠部分施加正则化。
    """
    penalty = tf.constant(0.0)
    for old_params, fisher in zip(old_params_list, fisher_list):
        for j, v in enumerate(tvars):
            v_old = old_params[j]
            f_old = fisher[j]
            # 形状可能因分类器扩展而变大——仅约束重叠区域
            if v.shape == v_old.shape:
                penalty += tf.reduce_sum(f_old * (v - v_old) ** 2)
            elif v.shape.ndims == 2 and v.shape[0] == v_old.shape[0]:
                # 分类器 kernel: (dim, new_classes) vs (dim, old_classes)
                old_cols = v_old.shape[1]
                penalty += tf.reduce_sum(
                    f_old * (v[:, :old_cols] - v_old) ** 2)
            elif v.shape.ndims == 1 and v.shape[0] > v_old.shape[0]:
                # bias / 1D variable
                old_len = v_old.shape[0]
                penalty += tf.reduce_sum(
                    f_old * (v[:old_len] - v_old) ** 2)
    return 0.5 * EWC_LAMBDA * penalty


def snapshot_params(tvars):
    """快照当前参数值。返回与 tvars 同序的 numpy 列表。"""
    return [v.numpy().copy() for v in tvars]


# ────────────────────────────────────────────────────────────
# 训练
# ────────────────────────────────────────────────────────────

def ewc_train_base(feature_extractor, classifier, X_train_cnn, y_train_mapped,
                   epochs, lr):
    """基类训练: 标准 CE（EWC 原论文方式）。"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    n = len(X_train_cnn)
    for epoch in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED + epoch).permutation(n)
        loss_sum, nb = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            bi = idx[i:i + BATCH_SIZE]
            Xb = tf.constant(X_train_cnn[bi])
            yb = tf.constant(y_train_mapped[bi])
            with tf.GradientTape() as tape:
                logits = classifier(feature_extractor(Xb, training=True))
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=yb, logits=logits))
            tvars = feature_extractor.trainable_variables + classifier.trainable_variables
            grads = tape.gradient(loss, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
            loss_sum += float(loss); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | CE: {loss_sum / max(nb, 1):.4f}")


def ewc_train_incremental(feature_extractor, classifier, X_new_cnn, y_new_mapped,
                          epochs, lr, old_params_list, fisher_list):
    """EWC 增量训练: CE + EWC 正则化（无回放、无蒸馏）。"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=INC_WEIGHT_DECAY)
    n = len(X_new_cnn)

    for epoch in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED + epoch).permutation(n)
        total_ce = total_ewc = 0.0
        nb = 0
        for i in range(0, n, INC_BATCH_SIZE):
            bi = idx[i:i + INC_BATCH_SIZE]
            Xb = tf.constant(X_new_cnn[bi])
            yb = tf.constant(y_new_mapped[bi])
            with tf.GradientTape() as tape:
                logits = classifier(feature_extractor(Xb, training=True))
                loss_ce = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=yb, logits=logits))
                loss_total = loss_ce
                if old_params_list:
                    tvars = feature_extractor.trainable_variables + classifier.trainable_variables
                    loss_total += ewc_penalty(tvars, old_params_list, fisher_list)
            tvars = feature_extractor.trainable_variables + classifier.trainable_variables
            grads = tape.gradient(loss_total, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
            total_ce += float(loss_ce)
            total_ewc += float(loss_total - loss_ce)
            nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | CE: {total_ce / max(nb, 1):.4f} "
                  f"| EWC: {total_ewc / max(nb, 1):.4f}")


# ────────────────────────────────────────────────────────────
# 评估
# ────────────────────────────────────────────────────────────

def evaluate_osr(feature_extractor, classifier, X_known, y_known, X_unknown, label_map):
    feats_known = feature_extractor.predict(X_known[..., np.newaxis], verbose=0)
    feats_unknown = feature_extractor.predict(X_unknown[..., np.newaxis], verbose=0)
    y_known_mapped = np.array([label_map[g] for g in y_known])

    logits_k = classifier(tf.constant(feats_known)).numpy()
    logits_u = classifier(tf.constant(feats_unknown)).numpy()
    probs_k = tf.nn.softmax(logits_k).numpy()
    probs_u = tf.nn.softmax(logits_u).numpy()

    scores_known = np.max(probs_k, axis=1)
    scores_unknown = np.max(probs_u, axis=1)
    preds_known = np.argmax(logits_k, axis=1)
    known_cls_acc = float(np.mean(preds_known == y_known_mapped))

    all_scores = np.concatenate([scores_known, scores_unknown])
    y_true = np.concatenate([np.ones(len(scores_known)), np.zeros(len(scores_unknown))])
    try:
        auroc = float(roc_auc_score(y_true, all_scores))
    except ValueError:
        auroc = None

    thresholds = sorted(np.unique(np.concatenate([
        np.percentile(all_scores, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]),
        np.linspace(all_scores.min(), all_scores.max(), 6)])))

    sweep = []
    nk, nu = len(scores_known), len(scores_unknown)
    for t in thresholds:
        tp = int(np.sum(scores_known >= t)); tn = int(np.sum(scores_unknown < t))
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
    logits = classifier(tf.constant(feats)).numpy()
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
    print("  EWC 对比实验 (Elastic Weight Consolidation)")
    print("  基类训练: 标准 CE（原论文方式）")
    print("  防遗忘: Fisher 信息正则化 (无回放)")
    print("  λ =", EWC_LAMBDA)
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
    y0_train_m = np.array([label_map[v] for v in y0_train])
    X0_cnn = X0_train[..., np.newaxis]

    feat_dim = feature_extractor(X0_cnn[:1]).shape[-1]
    classifier = DynamicClassifier(feat_dim, num_known)

    print(f"\n基类训练: {num_known} 类 (ArcFace)")
    ewc_train_base(feature_extractor, classifier, X0_cnn, y0_train_m,
                   EPOCHS_BASE, config.LEARNING_RATE)

    # 计算基类 Fisher + 快照参数
    print("  计算 Fisher 信息...")
    tvars = feature_extractor.trainable_variables + classifier.trainable_variables
    fisher0, tvars0 = compute_fisher(feature_extractor, classifier,
                             X0_cnn[..., np.newaxis], y0_train_m)
    params0 = snapshot_params(tvars)
    old_params_list = [params0]
    fisher_list = [fisher0]

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
            'best_g_mean': round(osr0['best_g_mean'], 4)
                if osr0['best_g_mean'] else None,
            'best_g_threshold': osr0['best_g_threshold'],
            'threshold_sweep': osr0['threshold_sweep']}}]
    per_class_best = dict(r0.get('per_class_accuracy', {}))

    for task_id in range(1, len(data.inc_batches) + 1):
        print(f"\n{'─' * 50}\n  增量步骤 {task_id}")
        X_new, y_new = data.get_task_data(task_id, 'train')
        X_new_cnn = X_new[..., np.newaxis]
        new_classes = sorted(np.unique(y_new))
        for gid in new_classes:
            if gid not in label_map:
                label_map[gid] = num_known; num_known += 1
        y_new_mapped = np.array([label_map[v] for v in y_new])

        classifier.add_classes(len(new_classes))
        print(f"  分类器: {classifier.num_classes} 输出节点")

        ewc_train_incremental(feature_extractor, classifier, X_new_cnn, y_new_mapped,
                              EPOCHS_INCREMENTAL, INC_LEARNING_RATE,
                              old_params_list, fisher_list)

        # 计算本任务 Fisher + 快照参数（累加到 EWC 约束）
        fisher_t, _ = compute_fisher(feature_extractor, classifier,
                                  X_new_cnn[..., np.newaxis], y_new_mapped)
        tvars = feature_extractor.trainable_variables + classifier.trainable_variables
        params_t = snapshot_params(tvars)
        old_params_list.append(params_t)
        fisher_list.append(fisher_t)

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
        stages.append({'task_id': task_id, 'stage': f'Task {task_id} (+{len(new_classes)}类)',
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
                'best_g_mean': round(osr['best_g_mean'], 4)
                    if osr['best_g_mean'] else None,
                'best_g_threshold': osr['best_g_threshold'],
                'threshold_sweep': osr['threshold_sweep']}})
        print(f"  CIL: {r['accuracy']:.4f}, 新类: {new_acc:.4f}, "
              f"遗忘: {avg_forg:.4f}, AUROC: {osr['confidence_auroc']}, "
              f"G-mean: {osr['best_g_mean']}")

    final_acc = stages[-1]['cil']['accuracy']
    avg_forg_total = float(np.mean([s['cil']['avg_forgetting'] for s in stages[1:]]))
    overall_avg = float(np.mean([s['cil']['accuracy'] for s in stages]))
    results = {'model': 'EWC (ArcFace + Fisher, w/o Replay)',
        'description': f'ArcFace分类 + EWC正则化(λ={EWC_LAMBDA})；无回放无蒸馏',
        'summary': {'final_accuracy': round(final_acc, 4),
            'avg_total_forgetting': round(avg_forg_total, 4),
            'overall_avg_accuracy': round(overall_avg, 4),
            'accuracy_drop': round(stages[0]['cil']['accuracy'] - final_acc, 4)},
        'stages': stages}

    print(f"\n{'=' * 64}\n  EWC 结果")
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
