# OPAL-LIBS

基于激光诱导击穿光谱（LIBS, Laser-Induced Breakdown Spectroscopy）的刀具**开集识别 + 类增量学习**（Open-Set Recognition & Class-Incremental Learning）系统。

OPAL 为本文提出的联合方法，在增量学习框架内同时解决「旧知识遗忘」与「未知类别拒识」两个问题，核心组件包括：

- **ArcFace 边距损失**：在角度空间施加加性边距，使同类特征聚集、异类特征远离，同步提升闭集判别与开集区分能力；
- **监督对比学习（SupCon）**：投影头 + 类别平衡采样，进一步收紧类内分布；
- **范例回放 + 关系蒸馏**：缓解类增量学习中的灾难性遗忘；
- **纯置信度开集检测**：以 `max(softmax(ArcFace logits))` 与阈值比较实现未知类拒识。

---

## 目录结构

```
OPAL-LIBS/
├── main.py                 # OPAL 模型入口（基类训练 + 增量学习 + 开集识别）
├── src/                    # 核心代码
│   ├── config.py           # 超参数与路径配置
│   ├── models.py           # 1D-CNN 特征提取器 + 可动态扩展分类器
│   ├── data_manager.py     # LIBS 数据加载与基类/增量类/未知类划分
│   ├── losses.py           # ArcFace 损失 + SupCon 损失
│   ├── incremental.py      # 增量学习器（范例回放 + 关系蒸馏）
│   ├── evaluator.py        # 闭集 / 开集 / 增量全阶段评估
│   ├── metrics.py          # 指标追踪（遗忘率、混淆矩阵、特征空间分析）
│   └── trainer.py          # OSR_CIL_Trainer 训练器
├── comparison/             # 对比实验基线模型
│   ├── MSP.py              # Maximum Softmax Probability
│   ├── OpenMax.py          # OpenMax（Weibull 尾部建模）
│   ├── CROSR.py            # CROSR（重构辅助开集识别）
│   ├── iCaRL.py            # iCaRL（最近类均值 + 蒸馏）
│   ├── LwF.py              # Learning without Forgetting
│   ├── EWC.py              # Elastic Weight Consolidation
│   └── DER++.py            # Dark Experience Replay
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 数据说明

**本仓库不提供数据集，仅提供代码。** 如需运行，请自行准备数据并放置为：

```
data/all_data.csv
```

数据格式要求：

- 每条样本为一段 LIBS 光谱，包含 **7062 维**光谱强度特征，最后一列为**类别标签**（整数）；
- 基类 / 增量类 / 未知类的划分比例由 `src/config.py` 中的 `BASE_RATIO`、`INC_RATIO`、`UNKNOWN_RATIO` 控制。

---

## 环境安装

建议使用 Python 3.10+ 与虚拟环境：

```bash
cd OPAL-LIBS
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

> 依赖包括 TensorFlow（主框架）、PyTorch（数据加载器导入）、scikit-learn、SciPy、pandas、matplotlib、seaborn 等。

---

## 快速开始

```bash
# 1. 将你的数据放到 data/all_data.csv（见上方「数据说明」）

# 2. 训练 OPAL 模型（基类 + 3 轮增量学习 + 开集识别评估）
python main.py
```

主要可调超参数位于 `src/config.py`（类别划分比例、ArcFace 的 `s`/`m`、SupCon 的 `λ`/`τ`、开集置信度阈值等）。

### 运行对比实验基线

```bash
python comparison/MSP.py
python comparison/OpenMax.py
python comparison/CROSR.py
python comparison/iCaRL.py
python comparison/LwF.py
python comparison/EWC.py
python comparison/DER++.py
```

各基线的结果以 JSON 形式保存至 `results/` 目录。

---

## License

本项目基于 [MIT License](LICENSE) 开源。
