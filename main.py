# main.py（放在项目根目录）

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import random
import numpy as np
import tensorflow as tf
from src.config import DATA_PATH, RANDOM_SEED
from src.data_manager import LIBSDataloader
from src.trainer import OSR_CIL_Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda', type=float, default=None, dest='lambda_supcon',
                        help='SupCon loss weight (overrides config)')
    parser.add_argument('--temp', type=float, default=None, dest='temperature',
                        help='SupCon temperature (overrides config)')
    parser.add_argument('--arcface-s', type=float, default=None, dest='arcface_s',
                        help='ArcFace scale factor s (overrides config, default 30.0)')
    parser.add_argument('--arcface-m', type=float, default=None, dest='arcface_m',
                        help='ArcFace angular margin m (overrides config, default 0.50)')
    parser.add_argument('--osr-conf', type=float, default=None, dest='osr_conf',
                        help='OSR softmax confidence threshold (overrides config, default 0.85)')
    args = parser.parse_args()

    # 命令行参数覆盖 config
    import src.config as config
    if args.lambda_supcon is not None:
        config.SUPCON_LAMBDA = args.lambda_supcon
    if args.temperature is not None:
        config.SUPCON_TEMPERATURE = args.temperature
    if args.arcface_s is not None:
        config.ARCFACE_S = args.arcface_s
    if args.arcface_m is not None:
        config.ARCFACE_M = args.arcface_m
    if args.osr_conf is not None:
        config.OSR_CONFIDENCE_THRESHOLD = args.osr_conf

    # ── 全局随机种子：确保所有随机过程可复现 ──
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()

    print("=" * 60)
    print("LIBS 刀具开集增量识别系统")
    print(f"SupCon λ={config.SUPCON_LAMBDA}, τ={config.SUPCON_TEMPERATURE}")
    print(f"ArcFace s={config.ARCFACE_S}, m={config.ARCFACE_M}")
    print(f"OSR: 纯置信度 (conf≥{config.OSR_CONFIDENCE_THRESHOLD})")
    print("=" * 60)

    # 1. 加载数据（Z-score 标准化 + PCA 保留 100% 方差）
    data = LIBSDataloader(DATA_PATH, random_seed=RANDOM_SEED)
    data.apply_pca(variance_ratio=config.PCA_VARIANCE, use_pca=True)

    # 2. 初始化训练器
    trainer = OSR_CIL_Trainer(data)

    # 3. 训练基类
    trainer.train_base()

    # 4. 增量学习
    for task_id in range(1, len(data.inc_batches) + 1):
        trainer.incremental_step(task_id)

    # 5. 增量学习指标汇总
    trainer.final_summary()

    # 6. 最终评估
    unknown_idx = data.get_indices_by_class(data.unknown_ids)
    print(f"\n未知类样本数: {len(unknown_idx)}")
    print("训练流程完成。")


if __name__ == "__main__":
    main()
