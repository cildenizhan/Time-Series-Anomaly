from sklearn.model_selection import GroupKFold
import numpy as np

def split_batadal(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df

def get_skab_cv(df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    groups = df['source_file'].values
    return gkf, groups
