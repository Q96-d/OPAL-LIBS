# src/metrics.py
"""
增量学习（Class Incremental Learning）指标追踪系统。

追踪的指标按阶段划分：

| 阶段              | 已知类别数 | 需要记录的指标             |
| --------------- | ----- | ------------------- |
| Task 0（基类）    | 80    | 初始准确率、混淆矩阵、特征空间结构 |
| Task 1（+5类）   | 85    | 新类学习准确率、旧类遗忘率、平均准确率 |
| Task 2（+5类）   | 90    | 新类学习准确率、旧类遗忘率、平均准确率 |
| Task 3（+5类）   | 95    | 最终准确率、总遗忘率、平均准确率    |

关键公式：
    - 遗忘率 f_k = max(0, a_k_best - a_k_current)
      a_k_best: 类别 k 被学习后的最佳准确率
      a_k_current: 当前任务的准确率
    - 平均遗忘率 = mean(f_k for all old classes)
    - 新类准确率 = accuracy on newly introduced classes only
    - 平均准确率 = accuracy across all known classes
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, silhouette_score
)
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from collections import defaultdict
import json
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 依赖
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OSR_CONFIDENCE_THRESHOLD

# 解决 matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


class IncrementalMetrics:
    """
    增量学习全阶段指标追踪器。

    用法:
        tracker = IncrementalMetrics(save_dir='results/')

        # Task 0
        tracker.record_task0(acc, cm, feature_stats)

        # Task 1
        tracker.record_incremental(
            task_id=1,
            new_class_ids=[...],
            old_class_ids=[...],
            per_class_acc={class_id: acc},
            confusion_mat=cm,
            feature_stats={...}
        )

        # 最终报告
        tracker.summary()
    """

    def __init__(self, save_dir='results/'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 每阶段记录
        self.records = []  # list of dict, 每项为一个阶段的完整指标 (CIL)

        # OSR 记录（每阶段一次）
        self.osr_records = []  # list of dict: {task_id, metrics}

        # 类别级别的历史最佳准确率（用于计算遗忘率）
        self.class_best_acc = {}  # class_id -> best_accuracy

        # 类别首次出现的任务编号
        self.class_intro_task = {}  # class_id -> task_id

    # ──────────────────────────────────────────────
    # 记录接口
    # ──────────────────────────────────────────────

    def record_task0(self, accuracy, confusion_mat, per_class_acc,
                     feature_stats=None):
        """
        记录基类（Task 0）指标。

        参数：
            accuracy: float, 基类整体准确率
            confusion_mat: np.ndarray, 混淆矩阵 (80×80)
            per_class_acc: dict, {class_id: accuracy}
            feature_stats: dict, 可选，特征空间结构指标
        """
        record = {
            'stage': 'Task 0 (基类)',
            'task_id': 0,
            'num_known_classes': len(per_class_acc),
            'num_new_classes': len(per_class_acc),
            'accuracy': accuracy,
            'confusion_matrix': confusion_mat,
            'per_class_accuracy': per_class_acc,
            'feature_space': feature_stats or {},
        }

        # 记录最佳准确率
        for cid, acc in per_class_acc.items():
            self.class_best_acc[cid] = acc
            self.class_intro_task[cid] = 0

        self.records.append(record)
        self._print_task0(record)
        return record

    def record_incremental(self, task_id, new_class_ids, old_class_ids,
                           per_class_acc, confusion_mat,
                           feature_stats=None):
        """
        记录增量步骤指标。

        参数：
            task_id: int, 1/2/3
            new_class_ids: list, 本阶段新引入的类别
            old_class_ids: list, 之前已学习的类别
            per_class_acc: dict, {class_id: accuracy}
            confusion_mat: np.ndarray, 混淆矩阵
            feature_stats: dict, 可选，特征空间结构指标
        """
        # ---- 新类准确率 ----
        new_acc = np.mean([per_class_acc[c] for c in new_class_ids]) if new_class_ids else 0.0

        # ---- 旧类遗忘率 ----
        forgetting_rates = {}
        for c in old_class_ids:
            best = self.class_best_acc.get(c, 0.0)
            current = per_class_acc.get(c, 0.0)
            f = max(0.0, best - current)
            forgetting_rates[c] = f

        avg_forgetting = np.mean(list(forgetting_rates.values())) if forgetting_rates else 0.0

        # ---- 平均准确率 ----
        avg_accuracy = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0

        # ---- 更新最佳准确率 ----
        for cid, acc in per_class_acc.items():
            if acc > self.class_best_acc.get(cid, 0.0):
                self.class_best_acc[cid] = acc
            if cid not in self.class_intro_task:
                self.class_intro_task[cid] = task_id

        record = {
            'stage': f'Task {task_id} (+{len(new_class_ids)}类)',
            'task_id': task_id,
            'num_known_classes': len(per_class_acc),
            'num_new_classes': len(new_class_ids),
            'num_old_classes': len(old_class_ids),
            'new_class_accuracy': new_acc,
            'avg_forgetting': avg_forgetting,
            'forgetting_rates': forgetting_rates,
            'avg_accuracy': avg_accuracy,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': confusion_mat,
            'feature_space': feature_stats or {},
        }

        self.records.append(record)
        self._print_incremental(record)
        return record

    def record_osr(self, task_id, osr_metrics):
        """
        记录开集识别 (OSR) 评估结果。

        参数：
            task_id: int, 0/1/2/3
            osr_metrics: dict, 来自 evaluate_osr_confidence() 的输出
        """
        record = {
            'task_id': task_id,
            'metrics': osr_metrics,
        }
        self.osr_records.append(record)
        self._print_osr(task_id, osr_metrics)

    def _print_osr(self, task_id, metrics):
        """打印 OSR 评估详细信息 —— 三阶段独立评估"""
        tp = metrics.get('tp', 0)
        fn = metrics.get('fn', 0)
        fp = metrics.get('fp', 0)
        tn = metrics.get('tn', 0)

        print(f"\n{'═' * 66}")
        print(f"  Task {task_id} 开集识别 (OSR) 三阶段评估")
        print(f"{'═' * 66}")

        # ═══════════════════════════════════════════════════
        # 阶段 1：已知类测试集独立评估
        # ═══════════════════════════════════════════════════
        known_eval = metrics.get('known_eval', None)
        if known_eval is not None:
            ke = known_eval
            print(f"\n  ┌─ 阶段 1：已知类测试集独立评估 ─────────────────────")
            print(f"  │ 样本总数:           {ke['n_total']}")
            print(f"  │ 被接受为已知 (TP):  {ke['n_accepted']}")
            print(f"  │ 被误拒为未知 (FN):  {ke['n_rejected']}")
            print(f"  │ 已知类召回率:       {ke['known_recall']:.4f}  "
                  f"= TP/(TP+FN) = {ke['n_accepted']}/{ke['n_total']}")
            print(f"  │ 已知类分类准确率:   {ke['known_cls_acc']:.4f}  "
                  f"(仅针对被正确接受的 {ke['n_accepted']} 个样本)")
            print(f"  └{'─' * 54}")
        else:
            # 向后兼容：从顶层键构造
            n_known = metrics.get('n_known_total', tp + fn)
            print(f"\n  ┌─ 阶段 1：已知类测试集独立评估 ─────────────────────")
            print(f"  │ 样本总数:           {n_known}")
            print(f"  │ 被接受为已知 (TP):  {tp}")
            print(f"  │ 被误拒为未知 (FN):  {fn}")
            print(f"  │ 已知类召回率:       {metrics['known_recall']:.4f}  "
                  f"= TP/(TP+FN) = {tp}/{tp + fn}")
            print(f"  │ 已知类分类准确率:   {metrics['known_cls_acc']:.4f}  "
                  f"(仅针对被正确接受的 {tp} 个样本)")
            print(f"  └{'─' * 54}")

        # ═══════════════════════════════════════════════════
        # 阶段 2：未知类数据集独立评估
        # ═══════════════════════════════════════════════════
        unknown_eval = metrics.get('unknown_eval', None)
        if unknown_eval is not None:
            ue = unknown_eval
            print(f"\n  ┌─ 阶段 2：未知类数据集独立评估 ─────────────────────")
            print(f"  │ 样本总数:           {ue['n_total']}")
            print(f"  │ 正确拒绝 (TN):      {ue['n_rejected']}")
            print(f"  │ 误接受为已知 (FP):  {ue['n_accepted']}")
            print(f"  │ 未知类召回率:       {ue['unknown_recall']:.4f}  "
                  f"= TN/(TN+FP) = {ue['n_rejected']}/{ue['n_total']}")
            print(f"  └{'─' * 54}")
        else:
            n_unknown = metrics.get('n_unknown_total', tn + fp)
            print(f"\n  ┌─ 阶段 2：未知类数据集独立评估 ─────────────────────")
            print(f"  │ 样本总数:           {n_unknown}")
            print(f"  │ 正确拒绝 (TN):      {tn}")
            print(f"  │ 误接受为已知 (FP):  {fp}")
            print(f"  │ 未知类召回率:       {metrics['unknown_recall']:.4f}  "
                  f"= TN/(TN+FP) = {tn}/{tn + fp}")
            print(f"  └{'─' * 54}")

        # ═══════════════════════════════════════════════════
        # 阶段 3：混合数据集综合评估
        # ═══════════════════════════════════════════════════
        mixed_eval = metrics.get('mixed_eval', None)
        if mixed_eval is not None:
            me = mixed_eval
            print(f"\n  ┌─ 阶段 3：混合数据集综合评估 ───────────────────────")
            print(f"  │ 已知类样本数:       {me['n_known']}")
            print(f"  │ 未知类样本数:       {me['n_unknown']}")
            print(f"  │ 混合总样本数:       {me['n_total']}")
            print(f"  │")
            print(f"  │ 混淆矩阵（混合数据集）:")
            print(f"  │             预测未知  预测已知")
            print(f"  │   真实未知  {me['tn']:>6d}     {me['fp']:>6d}")
            print(f"  │   真实已知  {me['fn']:>6d}     {me['tp']:>6d}")
            print(f"  │")
            print(f"  │ 混合数据集综合准确率:  {me['overall_acc']:.4f}  "
                  f"= (TP+TN)/Total = ({me['tp']}+{me['tn']})/{me['n_total']}")
            print(f"  │ 已知类召回率:          {me['known_recall']:.4f}  = TP/(TP+FN)")
            print(f"  │ 已知类精确率:          {me['known_precision']:.4f}  = TP/(TP+FP)")
            print(f"  │ 未知类召回率:          {me['unknown_recall']:.4f}  = TN/(TN+FP)")
            f1_val = me.get('f1', None)
            if f1_val is not None:
                print(f"  │ F1 (已知类):           {f1_val:.4f}  = 2PR/(P+R)")
            print(f"  └{'─' * 54}")
        else:
            n_total = tp + tn + fp + fn
            n_correct = tp + tn
            print(f"\n  ┌─ 阶段 3：混合数据集综合评估 ───────────────────────")
            print(f"  │ 已知类样本数:       {tp + fn}")
            print(f"  │ 未知类样本数:       {tn + fp}")
            print(f"  │ 混合总样本数:       {n_total}")
            print(f"  │")
            print(f"  │ 混淆矩阵（混合数据集）:")
            print(f"  │             预测未知  预测已知")
            print(f"  │   真实未知  {tn:>6d}     {fp:>6d}")
            print(f"  │   真实已知  {fn:>6d}     {tp:>6d}")
            print(f"  │")
            print(f"  │ 混合数据集综合准确率:  {metrics['overall_acc']:.4f}  "
                  f"= (TP+TN)/Total = ({tp}+{tn})/{n_total}")
            print(f"  │ 已知类召回率:          {metrics['known_recall']:.4f}  = TP/(TP+FN)")
            print(f"  │ 已知类精确率:          {metrics['known_precision']:.4f}  = TP/(TP+FP)")
            print(f"  │ 未知类召回率:          {metrics['unknown_recall']:.4f}  = TN/(TN+FP)")
            f1_val = metrics.get('f1', None)
            if f1_val is not None:
                print(f"  │ F1 (已知类):           {f1_val:.4f}  = 2PR/(P+R)")
            print(f"  └{'─' * 54}")

        # ═══════════════════════════════════════════════════
        # 阶段 4：AUROC（阈值无关判别力）
        # ═══════════════════════════════════════════════════
        auroc = metrics.get('auroc', None)
        if auroc is not None:
            print(f"\n  ┌─ 阶段 4：AUROC (阈值无关判别力) ─────────────────────")
            ca = auroc.get('confidence_auroc')
            if ca is not None:
                quality = '优秀' if ca >= 0.90 else ('良好' if ca >= 0.80 else ('一般' if ca >= 0.70 else '较差'))
                print(f"  │ 置信度 AUROC:        {ca:.4f}  ({quality})")
            print(f"  └{'─' * 54}")

        # ═══════════════════════════════════════════════════
        # 阶段 5：置信度阈值扫参
        # ═══════════════════════════════════════════════════
        sweep = metrics.get('threshold_sweep', None)
        if sweep is not None:
            print(f"\n  ┌─ 阶段 5：置信度阈值扫参 ───────────────────────────")
            print(f"  │ {'阈值':>6s}  {'known_recall':>13s}  {'unknown_recall':>15s}  {'f1':>6s}")
            print(f"  │ {'─' * 6}  {'─' * 13}  {'─' * 15}  {'─' * 6}")
            current_t = OSR_CONFIDENCE_THRESHOLD
            best_t, best_f1 = None, -1
            for s in sweep:
                marker = ' ← 当前' if abs(s['threshold'] - current_t) < 0.001 else ''
                print(f"  │ {s['threshold']:>5.2f}   {s['known_recall']:>11.4f}   {s['unknown_recall']:>13.4f}   {s['f1']:>6.4f}{marker}")
                if s['f1'] > best_f1:
                    best_f1 = s['f1']
                    best_t = s['threshold']
            print(f"  │")
            print(f"  │ 最优 F1 阈值: {best_t:.2f} (F1={best_f1:.4f})")
            print(f"  └{'─' * 54}")

        print(f"{'═' * 66}\n")

    # ──────────────────────────────────────────────
    # 最终指标计算
    # ──────────────────────────────────────────────

    def final_metrics(self):
        """计算最终阶段的汇总指标"""
        if len(self.records) < 2:
            return {}

        last = self.records[-1]
        first = self.records[0]

        # 总遗忘率：最终阶段所有旧类（非 Task 0 引入的类）的平均遗忘
        old_class_ids = [c for c, t in self.class_intro_task.items() if t < last['task_id']]
        total_forgetting_rates = {}
        for c in old_class_ids:
            best = self.class_best_acc.get(c, 0.0)
            current = last['per_class_accuracy'].get(c, 0.0)
            total_forgetting_rates[c] = max(0.0, best - current)

        avg_total_forgetting = np.mean(list(total_forgetting_rates.values())) if total_forgetting_rates else 0.0

        # 最终准确率
        final_accuracy = last['avg_accuracy']

        # 所有阶段的平均准确率
        all_avg_acc = [r['accuracy'] if r['task_id'] == 0 else r['avg_accuracy'] for r in self.records]
        overall_avg_accuracy = np.mean(all_avg_acc)

        # 准确率下降（最终 vs 初始）
        accuracy_drop = first['accuracy'] - final_accuracy

        return {
            'final_accuracy': final_accuracy,
            'avg_total_forgetting': avg_total_forgetting,
            'overall_avg_accuracy': overall_avg_accuracy,
            'accuracy_drop': accuracy_drop,
            'total_forgetting_rates': total_forgetting_rates,
        }

    # ──────────────────────────────────────────────
    # 摘要报告
    # ──────────────────────────────────────────────

    def summary(self):
        """打印并返回完整的增量学习指标摘要"""
        print("\n" + "=" * 70)
        print("  增量学习 (CIL) 全阶段指标摘要")
        print("=" * 70)

        # ---- 阶段表格 ----
        header = f"{'阶段':<18} {'已知类':<8} {'新类准确率':<12} {'旧类遗忘率':<12} {'平均准确率':<12}"
        print(header)
        print("-" * 62)

        for r in self.records:
            if r['task_id'] == 0:
                new_acc_str = "—"
                forget_str = "—"
                avg_str = f"{r['accuracy']:.4f}"
            else:
                new_acc_str = f"{r['new_class_accuracy']:.4f}"
                forget_str = f"{r['avg_forgetting']:.4f}"
                avg_str = f"{r['avg_accuracy']:.4f}"

            print(f"{r['stage']:<18} {r['num_known_classes']:<8} "
                  f"{new_acc_str:<12} {forget_str:<12} {avg_str:<12}")

        # ---- 最终指标 ----
        final = self.final_metrics()
        if final:
            print("\n" + "-" * 62)
            print(f"  最终准确率:       {final['final_accuracy']:.4f}")
            print(f"  总平均遗忘率:     {final['avg_total_forgetting']:.4f}")
            print(f"  全阶段平均准确率: {final['overall_avg_accuracy']:.4f}")
            print(f"  准确率下降:       {final['accuracy_drop']:.4f}")

        # ---- OSR 摘要表格 ----
        if self.osr_records:
            stage_names = {r['task_id']: r['stage'] for r in self.records}
            print("\n" + "-" * 75)
            print("  开集识别 (OSR) 指标摘要")
            print("-" * 75)
            header_osr = (f"{'阶段':<18} {'已知Recall':<12} {'已知ClsAcc':<12} "
                          f"{'已知Prec':<12} {'未知Recall':<12} {'整体Acc':<10}")
            print(header_osr)
            for osr_r in self.osr_records:
                m = osr_r['metrics']
                tid = osr_r['task_id']
                sname = stage_names.get(tid, f"Task {tid}")
                print(f"{sname:<18} {m['known_recall']:<12.4f} "
                      f"{m['known_cls_acc']:<12.4f} "
                      f"{m['known_precision']:<12.4f} "
                      f"{m['unknown_recall']:<12.4f} "
                      f"{m['overall_acc']:<10.4f}")

        print("=" * 70)

        # ---- 保存 ----
        self._save_results(final)

        return {
            'records': self.records,
            'final_metrics': final,
        }

    def _save_results(self, final_metrics):
        """保存 CIL + OSR 综合结果到 OSR+CIL.json"""
        # 构建 OSR 索引：{task_id: full_record}
        osr_index = {r['task_id']: r for r in self.osr_records}

        combined_stages = []
        for r in self.records:
            task_id = r['task_id']

            # ---- CIL 子结构 ----
            cil_dict = {
                'num_known_classes': r['num_known_classes'],
                'num_new_classes': r.get('num_new_classes', r['num_known_classes']),
            }
            if task_id == 0:
                cil_dict['accuracy'] = r['accuracy']
                cil_dict['per_class_accuracy'] = r['per_class_accuracy']
            else:
                cil_dict['num_old_classes'] = r.get('num_old_classes', 0)
                cil_dict['new_class_accuracy'] = r['new_class_accuracy']
                cil_dict['avg_forgetting'] = r['avg_forgetting']
                cil_dict['avg_accuracy'] = r['avg_accuracy']
                if 'forgetting_rates' in r:
                    cil_dict['forgetting_rates'] = r['forgetting_rates']
                cil_dict['per_class_accuracy'] = r['per_class_accuracy']
            cil_dict['feature_space'] = r.get('feature_space', {})

            # ---- OSR 子结构 ----
            osr_entry = osr_index.get(task_id, None)
            if osr_entry and 'metrics' in osr_entry:
                osr_metrics = osr_entry['metrics']
                osr_dict = {
                    # 三阶段评估
                    'known_eval': osr_metrics.get('known_eval', None),
                    'unknown_eval': osr_metrics.get('unknown_eval', None),
                    'mixed_eval': osr_metrics.get('mixed_eval', None),
                    # 顶层汇总
                    'known_recall': osr_metrics.get('known_recall'),
                    'known_precision': osr_metrics.get('known_precision'),
                    'known_cls_acc': osr_metrics.get('known_cls_acc'),
                    'unknown_recall': osr_metrics.get('unknown_recall'),
                    'overall_acc': osr_metrics.get('overall_acc'),
                    'confusion_matrix': {
                        'tp': osr_metrics.get('tp'),
                        'fn': osr_metrics.get('fn'),
                        'fp': osr_metrics.get('fp'),
                        'tn': osr_metrics.get('tn'),
                    },
                    'n_known_total': osr_metrics.get('n_known_total'),
                    'n_unknown_total': osr_metrics.get('n_unknown_total'),
                    'per_class_known_accuracy': osr_metrics.get('per_class_known_accuracy'),
                    # 阈值无关指标
                    'auroc': osr_metrics.get('auroc'),
                    # 双判据融合诊断
                    'fusion_details': osr_metrics.get('fusion_details'),
                    'threshold_sweep': osr_metrics.get('threshold_sweep'),
                }
            else:
                osr_dict = None

            stage_entry = {
                'task_id': task_id,
                'stage': r['stage'],
                'cil': cil_dict,
                'osr': osr_dict,
            }
            combined_stages.append(stage_entry)

        summary_dict = {
            'stages': combined_stages,
            'final_metrics': {k: v for k, v in final_metrics.items()
                              if k != 'total_forgetting_rates'},
            'class_best_acc': self.class_best_acc,
            'class_intro_task': self.class_intro_task,
        }

        path = os.path.join(self.save_dir, 'OSR+CIL.json')
        # 预处理：递归转换所有 numpy key/value → Python 原生类型
        summary_dict = _recursive_convert(summary_dict)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        print(f"\nCIL+OSR 综合指标已保存至: {path}")

        # 保存混淆矩阵为 PNG 图像
        for r in self.records:
            cm = r.get('confusion_matrix')
            if cm is not None:
                self._save_confusion_matrix_image(
                    cm, r['task_id'],
                    n_classes=r['num_known_classes']
                )

        # 保存 OSR 二元混淆矩阵图像
        for osr_r in self.osr_records:
            m = osr_r['metrics']
            tid = osr_r['task_id']
            # 构建 2×2 混淆矩阵: 行=真实(未知,已知), 列=预测(未知,已知)
            osr_cm = np.array([
                [m['tn'], m['fp']],
                [m['fn'], m['tp']],
            ])
            self._save_osr_confusion_matrix_image(osr_cm, tid)

    def _save_confusion_matrix_image(self, cm, task_id, n_classes):
        """将混淆矩阵保存为 PNG 图像"""
        # 对大类数的混淆矩阵调整图像尺寸
        figsize = max(6, n_classes * 0.15)
        plt.figure(figsize=(figsize, figsize * 0.85))
        sns.heatmap(cm, annot=(n_classes <= 20), fmt='d', cmap='Blues',
                    xticklabels=(n_classes <= 20),
                    yticklabels=(n_classes <= 20),
                    cbar=True)
        plt.title(f'混淆矩阵 — Task {task_id} ({n_classes} 类)')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        img_path = os.path.join(self.save_dir, f"confusion_matrix_task{task_id}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f"混淆矩阵图像已保存至: {img_path}")

    def _save_osr_confusion_matrix_image(self, cm, task_id):
        """将 OSR 二元混淆矩阵保存为 PNG 图像"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测未知', '预测已知'],
                    yticklabels=['真实未知', '真实已知'])
        plt.title(f'开集识别混淆矩阵 — Task {task_id}')
        plt.tight_layout()
        img_path = os.path.join(self.save_dir, f"osr_confusion_matrix_task{task_id}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f"OSR 混淆矩阵图像已保存至: {img_path}")

    # ──────────────────────────────────────────────
    # 打印辅助
    # ──────────────────────────────────────────────

    def _print_task0(self, record):
        print("\n" + "-" * 50)
        print(f"  [Task 0 基类评估]")
        print(f"  已知类别数: {record['num_known_classes']}")
        print(f"  初始准确率: {record['accuracy']:.4f}")
        fs = record.get('feature_space', {})
        if fs:
            print(f"  特征空间结构:")
            for k, v in fs.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        print("-" * 50)

    def _print_incremental(self, record):
        print("\n" + "-" * 50)
        print(f"  [{record['stage']} 评估]")
        print(f"  已知类别数:        {record['num_known_classes']}")
        print(f"  新类学习准确率:    {record['new_class_accuracy']:.4f}")
        print(f"  旧类平均遗忘率:    {record['avg_forgetting']:.4f}")
        print(f"  所有类平均准确率:  {record['avg_accuracy']:.4f}")
        print("-" * 50)


# ═══════════════════════════════════════════════════════════════
# 特征空间结构分析
# ═══════════════════════════════════════════════════════════════

def analyze_feature_space(features, labels, sample_limit=5000, random_seed=42):
    """
    分析特征空间的结构特性（基于弦距 / chord distance）。

    所有距离计算均在 L2 归一化后的单位球面上进行，因此欧氏距离等价于弦距：
        chord(x, y) = ||x - y||₂ = √(2(1 - cosθ))
    这与 ArcFace / SupCon 等角度 margin 损失的优化目标一致。

    参数：
        features: np.ndarray, shape (n_samples, feature_dim)，原始特征（无需预归一化）
        labels: np.ndarray, shape (n_samples,)
        sample_limit: int, 最大采样数（避免超大矩阵）
        random_seed: int, 随机种子（用于子采样）

    返回：
        dict，包含:
            - silhouette_score: 轮廓系数 [-1, 1]，越高类间分离越好（弦距版）
            - intra_class_dist_mean: 平均类内弦距（越小越紧凑）
            - inter_class_dist_mean: 平均类间弦距（越大越分散）
            - davies_bouldin_like: 简化的 DB 指数（弦距版，越小越好）
            - n_classes: 类别数
    """
    # 采样
    if len(features) > sample_limit:
        rng = np.random.RandomState(random_seed)
        idx = rng.choice(len(features), sample_limit, replace=False)
        features = features[idx]
        labels = labels[idx]

    # ═══ L2 归一化：投影到单位球面，使欧氏距离 = 弦距 ═══
    features = normalize(features, norm='l2')

    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    if n_classes < 2:
        return {
            'silhouette_score': None,
            'intra_class_dist_mean': 0.0,
            'inter_class_dist_mean': 0.0,
            'davies_bouldin_like': 0.0,
            'n_classes': n_classes,
        }

    # 轮廓系数（基于弦距，metric='euclidean' 在归一化后 = 弦距）
    try:
        sil = silhouette_score(features, labels, metric='euclidean',
                               sample_size=min(len(features), 2000))
    except Exception:
        sil = None

    # 类内弦距：每类样本到其球面中心（mean 后重归一化）的平均弦距
    intra_dists = []
    class_centers = {}
    for c in unique_classes:
        idx_c = np.where(labels == c)[0]
        class_feats = features[idx_c]
        # 球面中心：单位向量的均值再归一化到球面
        center_raw = np.mean(class_feats, axis=0)
        center = center_raw / (np.linalg.norm(center_raw) + 1e-12)
        class_centers[c] = center
        if len(class_feats) > 1:
            # 弦距 = L2 距离（归一化后等价）
            d = np.mean(np.linalg.norm(class_feats - center, axis=1))
            intra_dists.append(d)

    intra_mean = np.mean(intra_dists) if intra_dists else 0.0

    # 类间弦距：所有球面中心两两之间的平均弦距
    centers_arr = np.array(list(class_centers.values()))
    if len(centers_arr) > 1:
        inter_dists = cdist(centers_arr, centers_arr, metric='euclidean')
        # 只取上三角（排除对角线）
        triu_idx = np.triu_indices_from(inter_dists, k=1)
        inter_mean = np.mean(inter_dists[triu_idx])
    else:
        inter_mean = 0.0

    # 简化的 Davies-Bouldin 指数（弦距版）：intra / inter，越小越好
    db_like = 0.0
    if intra_mean > 0 and inter_mean > 0:
        db_like = intra_mean / inter_mean

    return {
        'silhouette_score': float(sil) if sil is not None else None,
        'intra_class_dist_mean': float(intra_mean),
        'inter_class_dist_mean': float(inter_mean),
        'davies_bouldin_like': float(db_like),
        'n_classes': int(n_classes),
    }


def compute_per_class_accuracy(y_true, y_pred, class_ids=None):
    """
    计算每个类别的准确率。

    参数：
        y_true: np.ndarray, 真实标签
        y_pred: np.ndarray, 预测标签
        class_ids: list, 要计算的类别列表（None = 所有出现的类别）

    返回：
        dict, {class_id: accuracy}
    """
    if class_ids is None:
        class_ids = np.unique(np.concatenate([y_true, y_pred]))

    per_class = {}
    for c in class_ids:
        mask = (y_true == c)
        if mask.sum() > 0:
            per_class[c] = np.mean(y_pred[mask] == c)
        else:
            per_class[c] = 0.0

    return per_class


# ═══════════════════════════════════════════════════════════════
# JSON 序列化辅助
# ═══════════════════════════════════════════════════════════════

def _to_native(obj):
    """将单个值转换为 Python 原生类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _recursive_convert(obj):
    """
    递归转换数据结构中所有 numpy 类型为 Python 原生类型。
    同时将 dict 的 key 从 numpy 类型转为原生类型。
    """
    if isinstance(obj, dict):
        return {_to_native(k): _recursive_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_recursive_convert(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _to_native(obj)


def _json_serializer(obj):
    """将 numpy 类型转换为 Python 原生类型（用作 json.dump 的 default）"""
    return _to_native(obj)
