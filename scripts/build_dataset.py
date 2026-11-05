# scripts/build_dataset.py
"""
将 data/ 目录下的 10 个 LIBS 数据文件合并为 data/all_data.csv。

数据文件说明：
  - dataset_101.csv ~ dataset_105.csv：水果刀数据，每个文件包含每把水果刀的 5 条 LIBS 数据
  - dataset_201.csv ~ dataset_205.csv：菜刀数据，每个文件包含每把菜刀的 5 条 LIBS 数据

每个文件的第一行为表头（光谱波长 + "Label"），合并时只保留一次表头。
生成的数据格式：前 7062 列为光谱强度，最后一列为类别标签。

用法：
    python scripts/build_dataset.py
"""

import glob
import os
import sys

import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
OUTPUT_PATH = os.path.join(DATA_DIR, "all_data.csv")


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "dataset_*.csv")))
    if not files:
        print("错误：未找到 data/dataset_*.csv 文件！", file=sys.stderr)
        sys.exit(1)

    frames = []
    for path in files:
        # header=0：第一行（波长 + "Label"）作为表头自动跳过
        df = pd.read_csv(path)
        print(f"  {os.path.basename(path)}: {df.shape[0]} 行 × {df.shape[1]} 列")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # 确保最后一列（标签）为整数
    label_col = combined.columns[-1]
    combined[label_col] = combined[label_col].astype(int)

    combined.to_csv(OUTPUT_PATH, index=False)

    n_features = combined.shape[1] - 1
    n_classes = combined[label_col].nunique()
    print(f"\n合并完成：{len(files)} 个文件 → {combined.shape[0]} 条光谱，"
          f"{n_features} 维特征，{n_classes} 个类别")
    print(f"输出文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
