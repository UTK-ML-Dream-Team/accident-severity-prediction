from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings


def viz_columns_corr(df: pd.DataFrame, cols_to_visualize: List[str]) -> None:
    df_ = df.copy()
    df_ = df_[cols_to_visualize].rename(lambda x: x[:20] + '..' if len(x) > 22 else x,
                                        axis='columns')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(data=df_.corr(), cmap='coolwarm', annot=True, fmt=".1f",
                annot_kws={'size': 16}, ax=ax)


def plot_bpnn_results(title: str, losses: List, test_accuracy: float, accuracies: List,
                      times: List, subsample: int = 1):
    warnings.filterwarnings("ignore")
    losses_ = losses[::subsample]
    losses_ = list(zip(*losses_))
    loss_names = []
    loss_values = []

    for loss in losses_:
        loss_name, values = list(zip(*loss))
        loss_names.append(loss_name[0])
        loss_values.append(values)

    accuracies_ = accuracies[::subsample]
    if times is not None:
        times_ = times[::subsample]
    x = np.arange(1, len(accuracies_) + 1)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    sup_title = f"{title}\nMax Train Accuracy: {max(accuracies) * 100:.4f} %\n"
    if test_accuracy is not None:
        sup_title += f"\nTest Accuracy: {test_accuracy * 100:.4f} %"
    fig.suptitle(sup_title)
    # Accuracies
    ax[0][0].plot(x, accuracies_)
    ax[0][0].set_title(f'Accuracies per epoch')
    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Accuracies (%)")
    ax[0][0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[0][0].set_yticklabels([0, 25., 50., 75., 100.])
    x_ticks = ax[0][0].get_xticks().tolist()
    ax[0][0].set_xticklabels([int(x_tick * subsample) for x_tick in x_ticks])
    ax[0][0].grid(True)
    # Times
    if times is not None:
        ax[0][1].plot(x, times_)
        ax[0][1].set_title(f'Times per epoch')
        ax[0][1].set_xlabel("Epoch")
        ax[0][1].set_ylabel("Second(s)")
        x_ticks = ax[0][1].get_xticks().tolist()
        ax[0][1].set_xticklabels([int(x_tick * subsample) for x_tick in x_ticks])
    ax[0][1].grid(True)
    for ind, (name, loss) in enumerate(zip(loss_names, loss_values)):
        ax[1][ind].plot(x, loss)
        ax[1][ind].set_title(f'{name} per epoch')
        ax[1][ind].set_xlabel("Epoch")
        ax[1][ind].set_ylabel(name)
        x_ticks = ax[1][ind].get_xticks().tolist()
        ax[1][ind].set_xticklabels([int(x_tick * subsample) for x_tick in x_ticks])
        ax[1][ind].grid(True)
    fig.tight_layout()
    make_space_above(ax, top_margin=1)
    fig.show()


def make_space_above(axes, top_margin=1):
    """ Increase figure size to make top_margin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1 - s.top) * h + top_margin
    fig.subplots_adjust(bottom=s.bottom * h / figh, top=1 - top_margin / figh)
    fig.set_figheight(figh)
