# src/data_manager.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
import os
from src.config import *

class LIBSDataloader:
    """LIBS 数据加载与划分管理器"""
    
    def __init__(self, data_path, random_seed=42):
        self.data_path = data_path
        self.random_seed = random_seed
        self.X = None
        self.y = None
        self.base_ids = []
        self.inc_batches = []
        self.unknown_ids = []
        self._load_data()
        self._split_roles()
    
    def _load_data(self):
        """加载 CSV 数据"""
        df = pd.read_csv(self.data_path)
        self.X = df.iloc[:, :-1].values.astype(np.float32)
        self.y = df.iloc[:, -1].values.astype(np.int64)
        print(f"加载 {len(self.X)} 条光谱，{self.X.shape[1]} 维")
    
    def _split_roles(self):
        """随机划分基类、增量类、未知类"""
        rng = np.random.RandomState(self.random_seed)
        classes = np.unique(self.y)
        shuffled = rng.permutation(classes)
        
        n_base = int(len(shuffled) * BASE_RATIO)
        n_inc = int(len(shuffled) * INC_RATIO)
        
        self.base_ids = shuffled[:n_base].tolist()
        inc_ids = shuffled[n_base:n_base+n_inc].tolist()
        self.unknown_ids = shuffled[n_base+n_inc:].tolist()
        
        # 分为多批
        batch_size = len(inc_ids) // INC_BATCHES
        self.inc_batches = [
            inc_ids[i:i+batch_size] 
            for i in range(0, len(inc_ids), batch_size)
        ]
        
        print(f"基类: {len(self.base_ids)}, 增量类: {len(inc_ids)}, 未知类: {len(self.unknown_ids)}")
    
    def get_indices_by_class(self, class_ids):
        """获取指定类别的所有样本索引"""
        mask = np.isin(self.y, class_ids)
        return np.where(mask)[0]
    
    def split_spectra(self, indices, train_ratio=0.68, val_ratio=0.16, test_ratio=0.16):
        """按光谱划分训练/验证/测试"""
        rng = np.random.RandomState(self.random_seed)
        shuffled = rng.permutation(indices)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (shuffled[:n_train].tolist(),
                shuffled[n_train:n_train+n_val].tolist(),
                shuffled[n_train+n_val:].tolist())
    
    def get_task_data(self, task_id, mode='train'):
        """
        获取指定任务的训练、验证或测试数据
        task_id: 0=基类, 1,2,3=增量批次
        mode: 'train'、'val' 或 'test'
              - 'test' 时包含所有已出现类别（用于增量评估）
        """
        if task_id == 0:
            class_ids = self.base_ids
        else:
            batch_idx = task_id - 1
            if batch_idx >= len(self.inc_batches):
                raise ValueError(f"任务 {task_id} 超出范围")
            # 测试时包含所有已出现类别
            if mode == 'test':
                all_ids = self.base_ids.copy()
                for i in range(batch_idx + 1):
                    all_ids.extend(self.inc_batches[i])
                class_ids = all_ids
            else:
                class_ids = self.inc_batches[batch_idx]

        indices = self.get_indices_by_class(class_ids)
        train_idx, val_idx, test_idx = self.split_spectra(indices)

        if mode == 'train':
            return self.X[train_idx], self.y[train_idx]
        elif mode == 'val':
            return self.X[val_idx], self.y[val_idx]
        else:
            return self.X[test_idx], self.y[test_idx]
    
    def get_unknown_data(self):
        """获取永久未知类数据（仅用于测试）"""
        idx = self.get_indices_by_class(self.unknown_ids)
        return self.X[idx], self.y[idx]

    def get_osr_unknown_data(self, task_id):
        """
        获取动态未知集数据 —— 开集增量标准评估协议。

        在 Task t 的 OSR 评估中：
          已知 = 基类 + 已完成学习的增量批次 (0~t)
          未知 = 未来增量批次 (t+1~T) + 永久未知类

        这迫使模型拒识所有"尚未出现"的类别，符合真实开放世界场景。

        参数:
            task_id: int, 0=基类, 1/2/3=增量批次
        返回:
            X_unknown, y_unknown
        """
        # 当前已知：基类 + 前 task_id 个增量批次
        # task_id=0 → 只有基类；task_id=1 → 基类 + batch0
        known_ids = set(self.base_ids)
        for i in range(task_id):
            known_ids.update(self.inc_batches[i])

        # 未知 = 所有类别 - 当前已知（含未来增量类 + 永久未知类）
        all_ids = set(np.unique(self.y))
        dynamic_unknown = list(all_ids - known_ids)

        print(f"  [动态未知] Task {task_id}: 已知 {len(known_ids)} 类, "
              f"未知 {len(dynamic_unknown)} 类 "
              f"(未来增量 {len(dynamic_unknown) - len(self.unknown_ids)} + "
              f"永久未知 {len(self.unknown_ids)})")

        idx = self.get_indices_by_class(dynamic_unknown)
        return self.X[idx], self.y[idx]
    
    def apply_pca(self, variance_ratio=None, use_pca=False):
        """
        数据预处理：Z-score 标准化（可选 PCA 降维）。

        只基于基类训练集拟合 scaler（以及可选的 PCA），避免数据泄漏。

        参数：
            variance_ratio: float, PCA 保留方差比例（仅在 use_pca=True 时生效）
            use_pca: bool, 是否启用 PCA 降维（默认 False，仅做 Z-score 标准化）

        返回:
            (X_processed, n_components) —— 处理后数据和特征维数
        """
        base_idx = self.get_indices_by_class(self.base_ids)
        base_train, _, _ = self.split_spectra(base_idx)  # train 是第一个返回值
        X_base_train = self.X[base_train]

        # ── Z-score 标准化（始终执行）──
        scaler = StandardScaler()
        X_base_scaled = scaler.fit_transform(X_base_train)
        joblib.dump(scaler, "scaler.pkl")

        # ── 可选 PCA ──
        if use_pca:
            if variance_ratio is None or variance_ratio >= 1.0:
                pca = PCA(n_components=None)
            else:
                pca = PCA(n_components=variance_ratio)
            pca.fit(X_base_scaled)
            joblib.dump(pca, "pca.pkl")

            X_all_scaled = scaler.transform(self.X)
            X_processed = pca.transform(X_all_scaled)
            n_components = pca.n_components_
            explained_var = np.sum(pca.explained_variance_ratio_)

            print(f"Z-score + PCA 完成:")
            print(f"  原始维度: {X_base_train.shape[1]}")
            print(f"  → 标准化 → PCA → {n_components} 维")
            print(f"  累计解释方差: {explained_var:.2%}")
        else:
            # 仅 Z-score 标准化，不降维
            X_processed = scaler.transform(self.X)
            n_components = X_processed.shape[1]

            print(f"Z-score 标准化完成:")
            print(f"  原始维度: {X_base_train.shape[1]}")
            print(f"  输出维度: {n_components}（无 PCA 降维）")
            print(f"  标准化: mean≈0, std≈1（按基类训练集统计量）")

        self.X = X_processed
        return X_processed, n_components