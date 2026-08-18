"""
房价预测调参实验

Assembled from your step-by-step solutions.
"""

# Step 2 - standardize_numeric
import pandas as pd
import numpy as np

def standardize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    for col in out.columns:
        s = out[col]
        mu = s.mean(skipna=True)
        sigma = s.std(skipna=True, ddof=1)
        if not np.isfinite(sigma) or sigma == 0:
            out[col] = 0.0
        else:
            out[col] = (s - mu) / sigma
    return out.fillna(0.0)

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
