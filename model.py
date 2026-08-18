"""
房价预测调参实验

Assembled from your step-by-step solutions.
"""

# Step 2 - standardize_numeric
import pandas as pd
import numpy as np

def standardize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    eps = np.finfo(float).eps
    for col in df_copy.columns:
        col_data = df_copy[col]
        mean_val = col_data.mean()
        std_val = col_data.std(ddof=0)
        # 全缺失则均值0，标准差1
        if col_data.isna().all():
            mean_val = 0
            std_val = 1
        else:
            # 标准差为0或非有限，替换为1
            if not np.isfinite(std_val) or std_val == 0:
                std_val = 1
            else:
                std_val = max(std_val, eps)
        # 标准化
        standardized = (col_data - mean_val) / std_val
        # NaN填充为0
        standardized = standardized.fillna(0)
        df_copy[col] = standardized
    return df_copy

# Step 3 - one_hot_encode
import pandas as pd

def one_hot_encode(cats: pd.Series, vocab: list) -> pd.DataFrame:
    """按 vocab 顺序独热编码，缺失当一类。"""
    n = len(cats)
    df = pd.DataFrame(0.0, index=cats.index, columns=vocab)
    for val in vocab:
        df[val] = (cats == val).astype(float)
    return df

# Step 4 - build_features (not yet solved)
# TODO: implement

# Step 5 - example_kernel (not yet solved)
# TODO: implement

# Step 6 - example_kernel (not yet solved)
# TODO: implement

# Step 7 - example_kernel (not yet solved)
# TODO: implement

# Step 8 - example_kernel (not yet solved)
# TODO: implement

# Step 9 - best_learning_rate (not yet solved)
# TODO: implement

# Step 10 - example_kernel (not yet solved)
# TODO: implement

# Step 11 - example_kernel (not yet solved)
# TODO: implement
