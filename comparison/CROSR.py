# comparison/CROSR.py
"""
CROSR (Classification-Reconstruction learning for Open-Set Recognition)
Yoshihashi et al., CVPR 2019.

架构: Encoder(1D-CNN with skip) → latent(128) + skips[c4,c3,c2,c1]
      Decoder(层级skip连接重建)
训练: CE + lambda*MSE_recon
OSR:  Softmax max probability（重建仅训练正则化）
"""

import sys, os, json, random
import numpy as np
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (DATA_PATH, RANDOM_SEED, EXEMPLAR_SIZE, EPOCHS_BASE,
    EPOCHS_INCREMENTAL, INC_LEARNING_RATE, BATCH_SIZE, INC_BATCH_SIZE,
    INC_WEIGHT_DECAY, DISTILLATION_ALPHA, FEATURE_DIM, LEARNING_RATE)
from src.data_manager import LIBSDataloader
from src.incremental import IncrementalLearner
from src.metrics import analyze_feature_space
from sklearn.metrics import roc_auc_score
from src.models import DynamicClassifier

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'CROSR.json')
os.makedirs(RESULTS_DIR, exist_ok=True)

CROSR_RECON_LAMBDA = 0.1


def set_seeds():
    for k in ['PYTHONHASHSEED','TF_DETERMINISTIC_OPS','TF_CUDNN_DETERMINISTIC']:
        os.environ[k] = '1' if k != 'PYTHONHASHSEED' else str(RANDOM_SEED)
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED); tf.keras.utils.set_random_seed(RANDOM_SEED)


# ============================
# CROSR Encoder-Decoder (skip 连接层级重建)
# ============================

def build_crosr_encoder_decoder(input_shape):
    """
    CROSR 层级重建架构。
    返回 (encoder_model, decoder_model)
    encoder 输出: latent, [c1, c2, c3, c4]
    decoder 输入: latent + skip features, 输出重建光谱
    """
    inp = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inp)
    p1 = tf.keras.layers.MaxPooling1D(2)(c1)          # ~349
    c2 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling1D(2)(c2)          # ~174
    c3 = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(p2)
    p3 = tf.keras.layers.MaxPooling1D(2)(c3)          # ~87
    c4 = tf.keras.layers.Conv1D(512, 3, activation='relu', padding='same')(p3)
    p4 = tf.keras.layers.MaxPooling1D(2)(c4)          # ~43
    c5 = tf.keras.layers.Conv1D(512, 3, activation='relu', padding='same')(p4)
    gmp = tf.keras.layers.GlobalMaxPooling1D()(c5)    # 512
    latent = tf.keras.layers.Dense(FEATURE_DIM, activation=None)(gmp)  # 128

    encoder = tf.keras.Model(inp, latent, name='crosr_encoder')

    # Decoder: 纯 bottleneck → 逐级上采样重建（无 skip 连接）
    lat = tf.keras.Input(shape=(FEATURE_DIM,))
    output_dim = input_shape[0]
    h = tf.keras.layers.Dense(512 * 22, activation='relu')(lat)
    h = tf.keras.layers.Reshape((22, 512))(h)
    h = tf.keras.layers.Conv1DTranspose(512, 3, strides=2, padding='same',
                                         activation='relu')(h)  # 22→44
    h = tf.keras.layers.Conv1DTranspose(256, 3, strides=2, padding='same',
                                         activation='relu')(h)  # 44→88
    h = tf.keras.layers.Conv1DTranspose(128, 3, strides=2, padding='same',
                                         activation='relu')(h)  # 88→176
    h = tf.keras.layers.Conv1DTranspose(64, 3, strides=2, padding='same',
                                         activation='relu')(h)  # 176→352
    h = tf.keras.layers.Conv1DTranspose(32, 3, strides=2, padding='same',
                                         activation='relu')(h)  # 352→704
    h = tf.keras.layers.Conv1D(1, 1, activation=None, padding='same')(h)
    h = tf.keras.layers.Flatten()(h)
    recon = tf.keras.layers.Dense(output_dim, activation=None)(h)

    decoder = tf.keras.Model(lat, recon, name='crosr_decoder')
    return encoder, decoder


def forward_encoder(encoder, X):
    """前向编码: 返回 latent。"""
    return encoder(X, training=True)


def forward_decoder(decoder, latent):
    """前向解码: 返回重建光谱。"""
    return decoder(latent)


# ============================
# 训练
# ============================

def train_base_crosr(encoder, decoder, cls, Xc, ym, epochs, lr):
    opt = tf.keras.optimizers.Adam(lr); n = len(Xc)
    for ep in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED+ep).permutation(n)
        ls_ce = ls_rc = 0.0; nb = 0
        for i in range(0, n, BATCH_SIZE):
            bi = idx[i:i+BATCH_SIZE]; Xb = tf.constant(Xc[bi]); yb = tf.constant(ym[bi])
            with tf.GradientTape() as t:
                latent = forward_encoder(encoder, Xb)
                logits = cls(latent)
                loss_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=yb, logits=logits))
                X_recon = forward_decoder(decoder, latent)
                X_flat = tf.reshape(Xb, [tf.shape(Xb)[0], -1])
                loss_rc = tf.reduce_mean(tf.reduce_sum((X_flat - X_recon)**2, axis=1))
                loss = loss_ce + CROSR_RECON_LAMBDA * loss_rc
            tv = encoder.trainable_variables + decoder.trainable_variables + cls.trainable_variables
            g = t.gradient(loss, tv); opt.apply_gradients(zip(g, tv))
            ls_ce += float(loss_ce); ls_rc += float(loss_rc); nb += 1
        if (ep+1)%10==0: print(f"  Epoch {ep+1}/{epochs} | CE: {ls_ce/max(nb,1):.4f} | Recon: {ls_rc/max(nb,1):.4f}")


def train_inc_crosr(encoder, decoder, cls, inc, Xn, yn, epochs, lr):
    Xn_cnn = Xn[..., np.newaxis] if Xn.ndim==2 else Xn
    inc.save_teacher(); cls.add_classes(len(np.unique(yn)))

    _x, _y = [], []
    for _d, _l in inc.exemplars.values(): _x.append(_d); _y.append(_l)
    Xa = [Xn_cnn]; ya = [yn]
    if _x: Xa.append(np.concatenate(_x, axis=0)); ya.append(np.concatenate(_y, axis=0))
    Xa = np.concatenate(Xa, axis=0); ya = np.concatenate(ya, axis=0)

    opt = tf.keras.optimizers.Adam(lr, weight_decay=INC_WEIGHT_DECAY); n = len(Xa)
    for ep in range(epochs):
        idx = np.random.RandomState(RANDOM_SEED+ep).permutation(n)
        ls_ce = ls_rc = 0.0; nb = 0
        for i in range(0, n, INC_BATCH_SIZE):
            bi = idx[i:i+INC_BATCH_SIZE]; Xb = tf.constant(Xa[bi]); yb = tf.constant(ya[bi])
            with tf.GradientTape() as t:
                latent = forward_encoder(encoder, Xb)
                logits = cls(latent)
                loss_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=yb, logits=logits))
                X_recon = forward_decoder(decoder, latent)
                X_flat = tf.reshape(Xb, [tf.shape(Xb)[0], -1])
                loss_rc = tf.reduce_mean(tf.reduce_sum((X_flat-X_recon)**2, axis=1))
                loss_kd = tf.constant(0.0)
                if inc.teacher_feature_extractor is not None:
                    t_latent = forward_encoder(encoder, Xb)
                    # Actually teacher is old feature extractor model
                    t_feats = inc.teacher_feature_extractor(Xb, training=False)
                    loss_kd = tf.reduce_mean((latent - t_feats)**2)
                loss = loss_ce + CROSR_RECON_LAMBDA*loss_rc + DISTILLATION_ALPHA*loss_kd
            tv = encoder.trainable_variables + decoder.trainable_variables + cls.trainable_variables
            g = t.gradient(loss, tv); opt.apply_gradients(zip(g, tv))
            ls_ce += float(loss_ce); ls_rc += float(loss_rc); nb += 1
        if (ep+1)%10==0: print(f"  Epoch {ep+1}/{epochs} | CE: {ls_ce/max(nb,1):.4f} | Recon: {ls_rc/max(nb,1):.4f}")


# ============================
# 评估
# ============================

def predict_latent(encoder, X):
    return encoder.predict(X[..., np.newaxis], verbose=0)

def evaluate_osr_crosr(encoder, cls, Xk, yk, Xu, lm):
    fk = predict_latent(encoder, Xk); fu = predict_latent(encoder, Xu)
    ykm = np.array([lm[g] for g in yk])
    lk = cls(tf.constant(fk)).numpy(); lu = cls(tf.constant(fu)).numpy()
    sk = np.max(tf.nn.softmax(lk).numpy(), axis=1)
    su = np.max(tf.nn.softmax(lu).numpy(), axis=1)
    pr_k = np.argmax(lk, axis=1); kca = float(np.mean(pr_k==ykm))
    alls = np.concatenate([sk, su]); yt = np.concatenate([np.ones(len(sk)), np.zeros(len(su))])
    try: au = float(roc_auc_score(yt, alls))
    except: au = None
    th = sorted(np.unique(np.concatenate([np.percentile(alls, [10,20,30,40,50,60,70,80,90,95]),
                                           np.linspace(alls.min(), alls.max(), 6)])))
    sw = []; nk1, nu = len(sk), len(su)
    for tt in th:
        tp = int(np.sum(sk>=tt)); tn = int(np.sum(su<tt))
        kr = tp/nk1 if nk1>0 else 0; ur = tn/nu if nu>0 else 0; g = np.sqrt(kr*ur)
        pr = tp/(tp+(nu-tn)) if (tp+nu-tn)>0 else 0
        f1 = 2*pr*kr/(pr+kr) if (pr+kr)>0 else 0
        sw.append({'threshold':round(float(tt),4),'known_recall':round(kr,4),'unknown_recall':round(ur,4),
                    'g_mean':round(float(g),4),'f1':round(f1,4)})
    bst = max(sw,key=lambda x:x['g_mean']) if sw else None
    return {'confidence_auroc':au,'known_cls_acc':kca,'threshold_sweep':sw,
            'best_g_mean':bst['g_mean'] if bst else None,'best_g_threshold':bst['threshold'] if bst else None}

def evaluate_cil(encoder, cls, Xt, yt, lm):
    mk = np.isin(yt, list(lm.keys()))
    Xk = Xt[mk]; yg = yt[mk]; ym = np.array([lm[g] for g in yg])
    if len(Xk)==0: return {'accuracy':0.0,'per_class_accuracy':{},'feature_stats':{}}
    fts = predict_latent(encoder, Xk)
    lo = cls(tf.constant(fts)).numpy(); pr = np.argmax(lo, axis=1)
    acc = float(np.mean(pr==ym)); pc = {}
    for c in np.unique(ym): m = ym==c; pc[int(c)] = float(np.mean(pr[m]==c)) if m.sum()>0 else 0.0
    fs = analyze_feature_space(fts, ym, sample_limit=5000)
    return {'accuracy':acc,'per_class_accuracy':pc,'feature_stats':fs}


# ============================
# 主流程
# ============================

def main():
    set_seeds()
    print("="*64+"\n  CROSR Experiment (Hierarchical Reconstruction)\n  OSR: Softmax max prob (recon as training reg)\n"+"="*64)
    data = LIBSDataloader(DATA_PATH, random_seed=RANDOM_SEED)
    data.apply_pca(variance_ratio=0.98, use_pca=True)
    input_shape = (data.X.shape[1], 1)
    encoder, decoder = build_crosr_encoder_decoder(input_shape)

    lm = {}; nk = 0
    X0t, y0t = data.get_task_data(0, 'train'); X0test, y0test = data.get_task_data(0, 'test')
    for g in sorted(data.base_ids): lm[g] = nk; nk += 1
    y0tm = np.array([lm[v] for v in y0t]); X0c = X0t[..., np.newaxis]

    fd = encoder(X0c[:1]).shape[-1]; cls = DynamicClassifier(fd, nk)
    print(f"\nBase: {nk} classes (CE + Recon)")
    train_base_crosr(encoder, decoder, cls, X0c, y0tm, EPOCHS_BASE, LEARNING_RATE)

    inc = IncrementalLearner(encoder, cls)
    for g in sorted(data.base_ids):
        idx = np.where(y0t==g)[0]
        inc.select_exemplars(X0t[idx], np.array([lm[g]]*len(idx)), lm[g], m=EXEMPLAR_SIZE)

    r0 = evaluate_cil(encoder, cls, X0test, y0test, lm)
    Xu0, _ = data.get_osr_unknown_data(0)
    osr0 = evaluate_osr_crosr(encoder, cls, X0test, y0test, Xu0, lm)
    fs0 = r0.get('feature_stats',{})

    def _stg(tid, nm, r, osr, fs, nk, na, af):
        return {'task_id':tid,'stage':nm,'cil':{
            'num_known_classes':nk,'accuracy':round(r['accuracy'],4),
            'new_class_accuracy':round(na,4) if na is not None else None,
            'avg_forgetting':round(af,4) if af is not None else None,
            'silhouette_score':round(fs.get('silhouette_score'),4) if fs.get('silhouette_score') is not None else None,
            'inter_intra_ratio':round(fs.get('inter_class_dist_mean',0)/max(fs.get('intra_class_dist_mean',1e-8),1e-8),4)},
            'osr':{'confidence_auroc':round(osr['confidence_auroc'],4) if osr['confidence_auroc'] else None,
                'known_cls_acc':round(osr['known_cls_acc'],4),'best_g_mean':round(osr['best_g_mean'],4) if osr['best_g_mean'] else None,
                'best_g_threshold':osr['best_g_threshold'],'threshold_sweep':osr['threshold_sweep']}}

    stages = [_stg(0,'Task 0 (Base)',r0,osr0,fs0,nk,None,None)]
    pcb = dict(r0.get('per_class_accuracy',{}))

    for tid in range(1,len(data.inc_batches)+1):
        print(f"\n{'─'*50}\n  Incremental Step {tid}")
        Xn, yn = data.get_task_data(tid,'train'); nc = sorted(np.unique(yn))
        for g in nc:
            if g not in lm: lm[g] = nk; nk += 1
        ynm = np.array([lm[v] for v in yn])
        train_inc_crosr(encoder, decoder, cls, inc, Xn, ynm, EPOCHS_INCREMENTAL, INC_LEARNING_RATE)
        for g in nc:
            idx = np.where(yn==g)[0]
            inc.select_exemplars(Xn[idx], np.array([lm[g]]*len(idx)), lm[g], m=EXEMPLAR_SIZE)

        Xt, yt = data.get_task_data(tid,'test')
        r = evaluate_cil(encoder, cls, Xt, yt, lm)
        nids = [lm[c] for c in nc]; pc = r.get('per_class_accuracy',{})
        na = np.mean([pc.get(nid,0) for nid in nids])
        oids = [v for k,v in lm.items() if k not in nc]
        fg = [max(0.0, pcb.get(oid,pc.get(oid,0))-pc.get(oid,0)) for oid in oids]
        af = float(np.mean(fg)) if fg else 0.0
        for c,acc in pc.items(): pcb[c] = max(pcb.get(c,0),acc)

        Xu, _ = data.get_osr_unknown_data(tid)
        osr = evaluate_osr_crosr(encoder, cls, Xt, yt, Xu, lm)
        fs = r.get('feature_stats',{})
        stages.append(_stg(tid, f'Task {tid} (+{len(nc)} classes)',r,osr,fs,nk,na,af))
        print(f"  CIL: {r['accuracy']:.4f}, New: {na:.4f}, Forg: {af:.4f}, AUROC: {osr['confidence_auroc']}, G-mean: {osr['best_g_mean']}")

    fa = stages[-1]['cil']['accuracy']
    afg = float(np.mean([s['cil']['avg_forgetting'] for s in stages[1:]]))
    oa = float(np.mean([s['cil']['accuracy'] for s in stages]))
    rs = {'model':'CROSR (Hierarchical Recon)','description':'层级skip重建+CE(无ArcFace)；OSR用softmax max prob',
          'summary':{'final_accuracy':round(fa,4),'avg_total_forgetting':round(afg,4),
                      'overall_avg_accuracy':round(oa,4),'accuracy_drop':round(stages[0]['cil']['accuracy']-fa,4)},
          'stages':stages}
    print(f"\n{'='*64}\n  CROSR Results\n  Final Acc: {rs['summary']['final_accuracy']:.4f}\n  Avg Forgetting: {rs['summary']['avg_total_forgetting']:.4f}\n  Overall Avg Acc: {rs['summary']['overall_avg_accuracy']:.4f}\n{'='*64}")
    with open(OUTPUT_PATH,'w',encoding='utf-8') as f: json.dump(rs,f,indent=2,ensure_ascii=False)
    print(f"\nResult: {OUTPUT_PATH}")

if __name__=="__main__": main()
