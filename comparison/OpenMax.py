# comparison/OpenMax.py
"""
OpenMax 对比基线 (Bendale & Boult, CVPR 2016).

替换: ArcFace → 标准 CE 分类器 + OpenMax OSR 层
保持: 范例回放 + 关系蒸馏（增量机制与 V3 一致，仅损失函数替换）

OpenMax OSR:
  1. 每类计算 MAV (Mean Activation Vector)
  2. 对每类尾部距离拟合 Weibull 分布
  3. 用 Weibull CDF 重校准 logits，增加"未知"类维度
  4. softmax 后取 max 为已知置信度，Unknown 概率为 OSR 得分
"""

import sys, os, json, random
import numpy as np
import tensorflow as tf
from scipy.stats import weibull_min

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.config import *
from src.models import build_feature_extractor, DynamicClassifier
from src.data_manager import LIBSDataloader
from src.incremental import IncrementalLearner
from src.metrics import analyze_feature_space
from sklearn.metrics import roc_auc_score, accuracy_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'OpenMax.json')
os.makedirs(RESULTS_DIR, exist_ok=True)

# OpenMax 超参数
OPENMAX_TAIL_SIZE = 5      # Weibull 尾部拟合样本数（≤ 范例数 12）
OPENMAX_ALPHA = 2.0        # 重校准缩放系数


def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED); tf.keras.utils.set_random_seed(RANDOM_SEED)


# ────────────────────────────────────────────────────────────
# OpenMax OSR 核心
# ────────────────────────────────────────────────────────────

def compute_mavs(feature_extractor, X, y_mapped, num_classes):
    """计算各类的 Mean Activation Vector (MAV)。X 已有通道维 (n, dim, 1)。"""
    feats = feature_extractor.predict(X, verbose=0)
    mavs = {}
    for c in range(num_classes):
        mask = y_mapped == c
        if mask.sum() > 0:
            mavs[c] = feats[mask].mean(axis=0)
        else:
            mavs[c] = np.zeros(feats.shape[1])
    return mavs


def fit_weibull_params(feature_extractor, X, y_mapped, mavs, num_classes,
                       tail_size=OPENMAX_TAIL_SIZE):
    """为每类拟合 Weibull 分布 (κ=shape, λ=scale)。X 已有通道维。"""
    feats = feature_extractor.predict(X, verbose=0)
    params = {}
    for c in range(num_classes):
        mask = y_mapped == c
        if mask.sum() < tail_size:
            params[c] = (1.0, 1.0)  # fallback
            continue
        dists = np.linalg.norm(feats[mask] - mavs[c], axis=1)
        tail = np.sort(dists)[-tail_size:]
        shape, _, scale = weibull_min.fit(tail, floc=0)
        params[c] = (shape, scale)
    return params


def openmax_scores(features, logits, mavs, weibull_params, num_classes,
                   alpha=OPENMAX_ALPHA):
    """
    OpenMax 重校准: 为已知类 + 1 个"未知"类输出得分。

    返回: scores (batch, K+1), confidence = softmax_max - P(unknown)
    """
    K = num_classes
    n = len(features)
    recal = np.zeros((n, K))

    for c in range(K):
        if c not in mavs or c not in weibull_params:
            recal[:, c] = logits[:, c]
            continue
        dists = np.linalg.norm(features - mavs[c], axis=1)
        kappa, lam = weibull_params[c]
        # Weibull CDF: 1 - exp(-(d/λ)^κ) = P(distance < d)
        # ω = 1 - CDF = exp(-(d/λ)^κ) = P(属于"尾部"") → 必须用 α 缩放
        omega = np.exp(-(alpha * dists / lam) ** kappa)
        recal[:, c] = logits[:, c] * omega

    # 未知类得分
    unknown_score = np.sum(logits - recal, axis=1, keepdims=True)
    scores = np.hstack([recal, unknown_score])

    # Softmax → 置信度
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    return probs


# ────────────────────────────────────────────────────────────
# 训练 (CE, 无 ArcFace)
# ────────────────────────────────────────────────────────────

def train_base_ce(fe, cls, Xc, ym, epochs, lr):
    opt = tf.keras.optimizers.Adam(lr); n = len(Xc)
    for ep in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED + ep).permutation(n)
        ls, nb = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            bi = idx[i:i + BATCH_SIZE]; Xb = tf.constant(Xc[bi]); yb = tf.constant(ym[bi])
            with tf.GradientTape() as t:
                lo = cls(fe(Xb, training=True))
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=yb, logits=lo))
            g = t.gradient(loss, fe.trainable_variables + cls.trainable_variables)
            opt.apply_gradients(zip(g, fe.trainable_variables + cls.trainable_variables))
            ls += float(loss); nb += 1
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep + 1}/{epochs} | CE: {ls / max(nb, 1):.4f}")


def train_inc_replay(fe, cls, inc, Xn, yn, epochs, lr):
    """增量训练: CE + 关系蒸馏 + 范例回放（V3 样，无 ArcFace）。"""
    Xn_cnn = Xn[..., np.newaxis]
    inc.incremental_train(
        Xn_cnn, yn,
        epochs=epochs, lr=lr,
        alpha=DISTILLATION_ALPHA,
        weight_decay=INC_WEIGHT_DECAY,
        projection_head=None,
        supcon_lambda=0.0)


# ────────────────────────────────────────────────────────────
# 评估
# ────────────────────────────────────────────────────────────

def evaluate_osr_openmax(fe, cls, mavs, wp, Xk, yk, Xu, lm):
    fk = fe.predict(Xk[..., np.newaxis], verbose=0)
    fu = fe.predict(Xu[..., np.newaxis], verbose=0)
    ykm = np.array([lm[g] for g in yk])
    nc = cls.num_classes

    lk = cls(tf.constant(fk)).numpy()
    lu = cls(tf.constant(fu)).numpy()
    pk = openmax_scores(fk, lk, mavs, wp, nc)
    pu = openmax_scores(fu, lu, mavs, wp, nc)

    # Known confidence = max over K known classes (not unknown dim)
    sk = np.max(pk[:, :nc], axis=1)
    su = np.max(pu[:, :nc], axis=1)
    pr_k = np.argmax(lk, axis=1)  # classification still uses standard logits
    kca = float(np.mean(pr_k == ykm))

    alls = np.concatenate([sk, su])
    yt = np.concatenate([np.ones(len(sk)), np.zeros(len(su))])
    try:
        au = float(roc_auc_score(yt, alls))
    except ValueError:
        au = None

    th = sorted(np.unique(np.concatenate(
        [np.percentile(alls, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]),
         np.linspace(alls.min(), alls.max(), 6)])))
    sw = []; nk, nu = len(sk), len(su)
    for tt in th:
        tp = int(np.sum(sk >= tt)); tn = int(np.sum(su < tt))
        kr = tp / nk if nk > 0 else 0; ur = tn / nu if nu > 0 else 0
        g = np.sqrt(kr * ur)
        pr = tp / (tp + (nu - tn)) if (tp + nu - tn) > 0 else 0
        f1 = 2 * pr * kr / (pr + kr) if (pr + kr) > 0 else 0
        sw.append({'threshold': round(float(tt), 4), 'known_recall': round(kr, 4),
                    'unknown_recall': round(ur, 4), 'g_mean': round(float(g), 4),
                    'f1': round(f1, 4)})
    bst = max(sw, key=lambda x: x['g_mean']) if sw else None
    return {'confidence_auroc': au, 'known_cls_acc': kca, 'threshold_sweep': sw,
            'best_g_mean': bst['g_mean'] if bst else None,
            'best_g_threshold': bst['threshold'] if bst else None}


def evaluate_cil_ce(fe, cls, Xt, yt, lm):
    mk = np.isin(yt, list(lm.keys()))
    Xk = Xt[mk]; yg = yt[mk]; ym = np.array([lm[g] for g in yg])
    if len(Xk) == 0:
        return {'accuracy': 0.0, 'per_class_accuracy': {}, 'feature_stats': {}}
    fts = fe.predict(Xk[..., np.newaxis], verbose=0)
    lo = cls(tf.constant(fts)).numpy(); pr = np.argmax(lo, axis=1)
    acc = float(np.mean(pr == ym))
    pc = {}
    for c in np.unique(ym):
        m = ym == c; pc[int(c)] = float(np.mean(pr[m] == c)) if m.sum() > 0 else 0.0
    fs = analyze_feature_space(fts, ym, sample_limit=5000)
    return {'accuracy': acc, 'per_class_accuracy': pc, 'feature_stats': fs}


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    set_seeds()
    print("=" * 64)
    print("  OpenMax Experiment")
    print("  分类: 标准 CE + Linear")
    print("  增量: 范例回放 + 关系蒸馏")
    print("  OSR:  OpenMax (Weibull + MAV)")
    print("=" * 64)

    data = LIBSDataloader(DATA_PATH, random_seed=RANDOM_SEED)
    data.apply_pca(variance_ratio=config.PCA_VARIANCE, use_pca=True)
    input_shape = (data.X.shape[1], 1)
    fe = build_feature_extractor(input_shape)

    lm = {}; nk = 0
    X0t, y0t = data.get_task_data(0, 'train'); X0test, y0test = data.get_task_data(0, 'test')
    for g in sorted(data.base_ids): lm[g] = nk; nk += 1
    y0tm = np.array([lm[v] for v in y0t]); X0c = X0t[..., np.newaxis]

    fd = fe(X0c[:1]).shape[-1]
    cls = DynamicClassifier(fd, nk)

    print(f"\nBase: {nk} classes (CE)")
    train_base_ce(fe, cls, X0c, y0tm, EPOCHS_BASE, config.LEARNING_RATE)

    # Init increment learner + exemplars
    inc = IncrementalLearner(fe, cls)
    for g in sorted(data.base_ids):
        idx = np.where(y0t == g)[0]
        inc.select_exemplars(X0t[idx], np.array([lm[g]] * len(idx)), lm[g], m=EXEMPLAR_SIZE)

    # OpenMax: compute MAVs + Weibull (仅用范例数据，遵守 CIL 内存约束)
    _x, _y = [], []
    for _cid, (_d, _l) in inc.exemplars.items():
        _x.append(_d); _y.append(_l)
    X_ex = np.concatenate(_x, axis=0); y_ex = np.concatenate(_y, axis=0)
    mavs = compute_mavs(fe, X_ex, y_ex, nk)
    wp = fit_weibull_params(fe, X_ex, y_ex, mavs, nk)

    r0 = evaluate_cil_ce(fe, cls, X0test, y0test, lm)
    Xu0, _ = data.get_osr_unknown_data(0)
    osr0 = evaluate_osr_openmax(fe, cls, mavs, wp, X0test, y0test, Xu0, lm)

    fs0 = r0.get('feature_stats', {})
    stages = [{'task_id': 0, 'stage': 'Task 0 (Base)', 'cil': {
        'num_known_classes': nk, 'accuracy': round(r0['accuracy'], 4),
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
    pcb = dict(r0.get('per_class_accuracy', {}))

    for tid in range(1, len(data.inc_batches) + 1):
        print(f"\n{'─' * 50}\n  Incremental Step {tid}")
        Xn, yn = data.get_task_data(tid, 'train'); Xn_cnn = Xn[..., np.newaxis]
        nc = sorted(np.unique(yn))
        for g in nc:
            if g not in lm: lm[g] = nk; nk += 1
        ynm = np.array([lm[v] for v in yn])

        # Save teacher for distillation
        inc.save_teacher()
        cls.add_classes(len(nc))
        print(f"  Classifier: {cls.num_classes} outputs")

        train_inc_replay(fe, cls, inc, Xn, ynm, EPOCHS_INCREMENTAL, INC_LEARNING_RATE)

        for g in nc:
            idx = np.where(yn == g)[0]
            inc.select_exemplars(Xn[idx], np.array([lm[g]] * len(idx)), lm[g], m=EXEMPLAR_SIZE)

        # Update OpenMax MAVs + Weibull（仅用范例数据，遵守 CIL 内存约束）
        _x, _y = [], []
        for _cid, (_d, _l) in inc.exemplars.items():
            _x.append(_d); _y.append(_l)
        X_ex = np.concatenate(_x, axis=0); y_ex = np.concatenate(_y, axis=0)
        mavs = compute_mavs(fe, X_ex, y_ex, nk)
        wp = fit_weibull_params(fe, X_ex, y_ex, mavs, nk)

        Xt, yt = data.get_task_data(tid, 'test')
        r = evaluate_cil_ce(fe, cls, Xt, yt, lm)
        nids = [lm[c] for c in nc]; pc = r.get('per_class_accuracy', {})
        na = np.mean([pc.get(nid, 0) for nid in nids])
        oids = [v for k, v in lm.items() if k not in nc]
        fg = [max(0.0, pcb.get(oid, pc.get(oid, 0)) - pc.get(oid, 0)) for oid in oids]
        af = float(np.mean(fg)) if fg else 0.0
        for c, acc in pc.items(): pcb[c] = max(pcb.get(c, 0), acc)

        Xu, _ = data.get_osr_unknown_data(tid)
        osr = evaluate_osr_openmax(fe, cls, mavs, wp, Xt, yt, Xu, lm)
        fs = r.get('feature_stats', {})
        stages.append({'task_id': tid, 'stage': f'Task {tid} (+{len(nc)} classes)',
            'cil': {'num_known_classes': nk, 'accuracy': round(r['accuracy'], 4),
                'new_class_accuracy': round(na, 4), 'avg_forgetting': round(af, 4),
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
        print(f"  CIL: {r['accuracy']:.4f}, New: {na:.4f}, "
              f"Forg: {af:.4f}, AUROC: {osr['confidence_auroc']}, "
              f"G-mean: {osr['best_g_mean']}")

    fa = stages[-1]['cil']['accuracy']
    afg = float(np.mean([s['cil']['avg_forgetting'] for s in stages[1:]]))
    oa = float(np.mean([s['cil']['accuracy'] for s in stages]))
    rs = {'model': 'OpenMax (CE + Replay + Distill)',
          'description': f'OpenMax OSR(Weibull tail=5, alpha={OPENMAX_ALPHA}); 增量同V3但CE替换ArcFace',
          'summary': {'final_accuracy': round(fa, 4),
                      'avg_total_forgetting': round(afg, 4),
                      'overall_avg_accuracy': round(oa, 4),
                      'accuracy_drop': round(stages[0]['cil']['accuracy'] - fa, 4)},
          'stages': stages}

    print(f"\n{'=' * 64}\n  OpenMax Results")
    s = rs['summary']
    print(f"  Final Acc:     {s['final_accuracy']:.4f}")
    print(f"  Avg Forgetting: {s['avg_total_forgetting']:.4f}")
    print(f"  Overall Avg:   {s['overall_avg_accuracy']:.4f}")
    print(f"{'=' * 64}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(rs, f, indent=2, ensure_ascii=False)
    print(f"\nResult: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
