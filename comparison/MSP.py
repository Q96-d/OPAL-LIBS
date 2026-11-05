# comparison/MSP.py
"""
MSP (Maximum Softmax Probability) 基线对比。
V3 去除 ArcFace，使用标准 CE + Softmax 置信度。

与 V3 的区别:
  - 分类器: 标准 Softmax（无 ArcFace margin、无 L2 归一化）
  - 损失: CE + SupCon（基类）/ CE + 关系蒸馏 + SupCon（增量）
  - OSR: max(softmax(logits))
  - 其余: 范例回放、关系蒸馏、SupCon 与 V3 完全一致
"""

import sys, os, json, random
import numpy as np
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.models import build_feature_extractor, DynamicClassifier
from src.data_manager import LIBSDataloader
from src.incremental import IncrementalLearner
from src.losses import supervised_contrastive_loss
from src.metrics import analyze_feature_space
from sklearn.metrics import roc_auc_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'MSP.json')
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED); tf.keras.utils.set_random_seed(RANDOM_SEED)


def _build_projection_head(input_dim):
    proj_input = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(proj_input)
    x = tf.keras.layers.Dense(PROJECTION_DIM)(x)
    return tf.keras.Model(proj_input, x, name='projection_head')


# ============================
# 训练
# ============================

def train_base_msp(fe, cls, proj, Xc, ym, epochs, lr):
    nk = cls.num_classes
    opt = tf.keras.optimizers.Adam(lr)
    batch_sz = SUPCON_CLASSES_PER_BATCH * SUPCON_SAMPLES_PER_CLASS
    unique = np.unique(ym)
    cidx = {c: np.where(ym == c)[0] for c in unique}

    def gen():
        rng = np.random.RandomState(RANDOM_SEED)
        while True:
            ch = rng.choice(unique, SUPCON_CLASSES_PER_BATCH, replace=False)
            bi = []
            for c in ch: bi.extend(rng.choice(cidx[c], SUPCON_SAMPLES_PER_CLASS, replace=True))
            rng.shuffle(bi)
            yield Xc[bi], ym[bi]

    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec(shape=(batch_sz, Xc.shape[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_sz,), dtype=tf.int64))).prefetch(tf.data.AUTOTUNE)
    steps = max(len(Xc) // batch_sz, 1)

    for ep in range(epochs):
        ls_ce = ls_sc = 0.0; nb = 0
        for Xb, yb in ds.take(steps):
            with tf.GradientTape() as t:
                feats = fe(Xb, training=True)
                logits = cls(feats)  # 标准 CE，无 ArcFace
                loss_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=yb, logits=logits))
                proj_out = proj(feats, training=True)
                proj_n = tf.math.l2_normalize(proj_out, axis=1)
                loss_sc = supervised_contrastive_loss(proj_n, yb, temperature=SUPCON_TEMPERATURE)
                loss = loss_ce + SUPCON_LAMBDA * loss_sc
            tv = fe.trainable_variables + cls.trainable_variables + proj.trainable_variables
            g = t.gradient(loss, tv); opt.apply_gradients(zip(g, tv))
            ls_ce += float(loss_ce); ls_sc += float(loss_sc); nb += 1
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep + 1}/{epochs} | CE: {ls_ce / max(nb, 1):.4f} "
                  f"| SupCon: {ls_sc / max(nb, 1):.4f}")


def train_inc_msp(fe, cls, inc, Xn, yn, epochs, lr):
    """增量: CE + 关系蒸馏 + 回放（与 V3 一致，仅无 ArcFace）。"""
    Xn_cnn = Xn[..., np.newaxis] if Xn.ndim == 2 else Xn
    inc.save_teacher()
    cls.add_classes(len(np.unique(yn)))

    _x, _y = [], []
    for _d, _l in inc.exemplars.values(): _x.append(_d); _y.append(_l)
    Xa = [Xn_cnn]; ya = [yn]
    if _x: Xa.append(np.concatenate(_x, axis=0)); ya.append(np.concatenate(_y, axis=0))
    Xa = np.concatenate(Xa, axis=0); ya = np.concatenate(ya, axis=0)

    opt = tf.keras.optimizers.Adam(lr, weight_decay=INC_WEIGHT_DECAY); n = len(Xa)
    for ep in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED + ep).permutation(n)
        ls_ce = ls_kd = 0.0; nb = 0
        for i in range(0, n, INC_BATCH_SIZE):
            bi = idx[i:i + INC_BATCH_SIZE]; Xb = tf.constant(Xa[bi]); yb = tf.constant(ya[bi])
            with tf.GradientTape() as t:
                feats = fe(Xb, training=True)
                logits = cls(feats)  # 标准 CE
                loss_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=yb, logits=logits))

                # 关系蒸馏（同 V3）
                loss_kd = tf.constant(0.0)
                if inc.teacher_feature_extractor is not None:
                    t_feats = inc.teacher_feature_extractor(Xb, training=False)
                    loss_kd = inc.relation_distillation_loss(feats, t_feats.numpy())

                loss = loss_ce + DISTILLATION_ALPHA * loss_kd
            tv = fe.trainable_variables + cls.trainable_variables
            g = t.gradient(loss, tv); opt.apply_gradients(zip(g, tv))
            ls_ce += float(loss_ce); ls_kd += float(loss_kd); nb += 1
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep + 1}/{epochs} | CE: {ls_ce / max(nb, 1):.4f} "
                  f"| KD: {ls_kd / max(nb, 1):.4f}")


# ============================
# 评估
# ============================

def evaluate_osr(fe, cls, Xk, yk, Xu, lm):
    fk = fe.predict(Xk[..., np.newaxis], verbose=0)
    fu = fe.predict(Xu[..., np.newaxis], verbose=0)
    ykm = np.array([lm[g] for g in yk])

    lk = cls(tf.constant(fk)).numpy(); lu = cls(tf.constant(fu)).numpy()
    sk = np.max(tf.nn.softmax(lk).numpy(), axis=1)
    su = np.max(tf.nn.softmax(lu).numpy(), axis=1)
    pk = np.argmax(lk, axis=1); kca = float(np.mean(pk == ykm))

    alls = np.concatenate([sk, su])
    yt = np.concatenate([np.ones(len(sk)), np.zeros(len(su))])
    try: au = float(roc_auc_score(yt, alls))
    except: au = None

    th = sorted(np.unique(np.concatenate(
        [np.percentile(alls, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]),
         np.linspace(alls.min(), alls.max(), 6)])))
    sw = []; nk1, nu = len(sk), len(su)
    for tt in th:
        tp = int(np.sum(sk >= tt)); tn = int(np.sum(su < tt))
        kr = tp / nk1 if nk1 > 0 else 0; ur = tn / nu if nu > 0 else 0; g = np.sqrt(kr * ur)
        pr = tp / (tp + (nu - tn)) if (tp + nu - tn) > 0 else 0
        f1 = 2 * pr * kr / (pr + kr) if (pr + kr) > 0 else 0
        sw.append({'threshold': round(float(tt), 4), 'known_recall': round(kr, 4),
                    'unknown_recall': round(ur, 4), 'g_mean': round(float(g), 4),
                    'f1': round(f1, 4)})
    bst = max(sw, key=lambda x: x['g_mean']) if sw else None
    return {'confidence_auroc': au, 'known_cls_acc': kca, 'threshold_sweep': sw,
            'best_g_mean': bst['g_mean'] if bst else None,
            'best_g_threshold': bst['threshold'] if bst else None}


def evaluate_cil(fe, cls, Xt, yt, lm):
    mk = np.isin(yt, list(lm.keys()))
    Xk = Xt[mk]; yg = yt[mk]; ym = np.array([lm[g] for g in yg])
    if len(Xk) == 0:
        return {'accuracy': 0.0, 'per_class_accuracy': {}, 'feature_stats': {}}
    fts = fe.predict(Xk[..., np.newaxis], verbose=0)
    lo = cls(tf.constant(fts)).numpy(); pr = np.argmax(lo, axis=1)
    acc = float(np.mean(pr == ym)); pc = {}
    for c in np.unique(ym):
        m = ym == c; pc[int(c)] = float(np.mean(pr[m] == c)) if m.sum() > 0 else 0.0
    fs = analyze_feature_space(fts, ym, sample_limit=5000)
    return {'accuracy': acc, 'per_class_accuracy': pc, 'feature_stats': fs}


# ============================
# 主流程
# ============================

def main():
    set_seeds()
    print("=" * 64)
    print("  MSP Baseline (CE + Softmax, no ArcFace)")
    print("  OSR: max(softmax(logits))")
    print("  Incremental: Replay + Relation Distill + SupCon")
    print("=" * 64)

    data = LIBSDataloader(DATA_PATH, random_seed=RANDOM_SEED)
    data.apply_pca(variance_ratio=PCA_VARIANCE, use_pca=True)
    input_shape = (data.X.shape[1], 1)
    fe = build_feature_extractor(input_shape)

    lm = {}; nk = 0
    X0t, y0t = data.get_task_data(0, 'train'); X0test, y0test = data.get_task_data(0, 'test')
    for g in sorted(data.base_ids): lm[g] = nk; nk += 1
    y0tm = np.array([lm[v] for v in y0t]); X0c = X0t[..., np.newaxis]

    fd = fe(X0c[:1]).shape[-1]; cls = DynamicClassifier(fd, nk)
    proj = _build_projection_head(fd)

    print(f"\nBase: {nk} classes (CE + SupCon)")
    train_base_msp(fe, cls, proj, X0c, y0tm, EPOCHS_BASE, LEARNING_RATE)

    inc = IncrementalLearner(fe, cls)
    for g in sorted(data.base_ids):
        idx = np.where(y0t == g)[0]
        inc.select_exemplars(X0t[idx], np.array([lm[g]] * len(idx)), lm[g], m=EXEMPLAR_SIZE)

    r0 = evaluate_cil(fe, cls, X0test, y0test, lm)
    Xu0, _ = data.get_osr_unknown_data(0)
    osr0 = evaluate_osr(fe, cls, X0test, y0test, Xu0, lm)
    fs0 = r0.get('feature_stats', {})

    def _stg(tid, nm, r, osr, fs, nk1, na, af):
        return {'task_id': tid, 'stage': nm, 'cil': {
            'num_known_classes': nk1, 'accuracy': round(r['accuracy'], 4),
            'new_class_accuracy': round(na, 4) if na is not None else None,
            'avg_forgetting': round(af, 4) if af is not None else None,
            'silhouette_score': round(fs.get('silhouette_score'), 4)
                if fs.get('silhouette_score') is not None else None,
            'inter_intra_ratio': round(fs.get('inter_class_dist_mean', 0) /
                max(fs.get('intra_class_dist_mean', 1e-8), 1e-8), 4)},
            'osr': {'confidence_auroc': round(osr['confidence_auroc'], 4)
                if osr['confidence_auroc'] else None,
                'known_cls_acc': round(osr['known_cls_acc'], 4),
                'best_g_mean': round(osr['best_g_mean'], 4) if osr['best_g_mean'] else None,
                'best_g_threshold': osr['best_g_threshold'],
                'threshold_sweep': osr['threshold_sweep']}}

    stages = [_stg(0, 'Task 0 (Base)', r0, osr0, fs0, nk, None, None)]
    pcb = dict(r0.get('per_class_accuracy', {}))

    for tid in range(1, len(data.inc_batches) + 1):
        print(f"\n{'─' * 50}\n  Incremental Step {tid}")
        Xn, yn = data.get_task_data(tid, 'train')
        nc = sorted(np.unique(yn)); nn = len(nc)
        for g in nc:
            if g not in lm: lm[g] = nk; nk += 1
        ynm = np.array([lm[v] for v in yn])
        train_inc_msp(fe, cls, inc, Xn, ynm, EPOCHS_INCREMENTAL, INC_LEARNING_RATE)
        for g in nc:
            idx = np.where(yn == g)[0]
            inc.select_exemplars(Xn[idx], np.array([lm[g]] * len(idx)), lm[g], m=EXEMPLAR_SIZE)

        Xt, yt = data.get_task_data(tid, 'test')
        r = evaluate_cil(fe, cls, Xt, yt, lm)
        nids = [lm[c] for c in nc]; pc = r.get('per_class_accuracy', {})
        na = np.mean([pc.get(nid, 0) for nid in nids])
        oids = [v for k, v in lm.items() if k not in nc]
        fg = [max(0.0, pcb.get(oid, pc.get(oid, 0)) - pc.get(oid, 0)) for oid in oids]
        af = float(np.mean(fg)) if fg else 0.0
        for c, acc in pc.items(): pcb[c] = max(pcb.get(c, 0), acc)

        Xu, _ = data.get_osr_unknown_data(tid)
        osr = evaluate_osr(fe, cls, Xt, yt, Xu, lm)
        fs = r.get('feature_stats', {})
        stages.append(_stg(tid, f'Task {tid} (+{nn} classes)', r, osr, fs, nk, na, af))
        print(f"  CIL: {r['accuracy']:.4f}, New: {na:.4f}, Forg: {af:.4f}, "
              f"AUROC: {osr['confidence_auroc']}, G-mean: {osr['best_g_mean']}")

    fa = stages[-1]['cil']['accuracy']
    afg = float(np.mean([s['cil']['avg_forgetting'] for s in stages[1:]]))
    oa = float(np.mean([s['cil']['accuracy'] for s in stages]))
    rs = {'model': 'MSP (CE + Softmax, no ArcFace)',
          'description': '标准CE+Softmax置信度；增量同V3(回放+关系蒸馏+SupCon)仅无ArcFace',
          'summary': {'final_accuracy': round(fa, 4), 'avg_total_forgetting': round(afg, 4),
                      'overall_avg_accuracy': round(oa, 4),
                      'accuracy_drop': round(stages[0]['cil']['accuracy'] - fa, 4)},
          'stages': stages}

    print(f"\n{'=' * 64}\n  MSP Results")
    s = rs['summary']
    print(f"  Final Acc:       {s['final_accuracy']:.4f}")
    print(f"  Avg Forgetting:  {s['avg_total_forgetting']:.4f}")
    print(f"  Overall Avg Acc: {s['overall_avg_accuracy']:.4f}")
    print(f"{'=' * 64}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(rs, f, indent=2, ensure_ascii=False)
    print(f"\nResult: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
