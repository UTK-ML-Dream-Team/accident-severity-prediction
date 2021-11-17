from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def viz_columns_corr(df: pd.DataFrame, cols_to_visualize: List[str]) -> None:
    df_ = df.copy()
    df_ = df_[cols_to_visualize].rename(lambda x: x[:20] + '..' if len(x) > 22 else x,
                                        axis='columns')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(data=df_.corr(), cmap='coolwarm', annot=True, fmt=".1f",
                annot_kws={'size': 16}, ax=ax)
