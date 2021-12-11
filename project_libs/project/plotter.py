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

def transform_data_for_plotting(accident_df):
    # Create duration feature
    accident_df['Start_Time'] = pd.to_datetime(accident_df['Start_Time'])
    accident_df['End_Time'] = pd.to_datetime(accident_df['End_Time'])
    accident_df['Duration'] = (accident_df['End_Time'] - accident_df['Start_Time']).dt.total_seconds() / 3600.0  # Hours

    # Club the target variables
    accident_df.loc[(accident_df['Severity'] == 1) | (accident_df['Severity'] == 2), 'Severity'] = 0
    accident_df.loc[(accident_df['Severity'] == 3) | (accident_df['Severity'] == 4), 'Severity'] = 1
    return accident_df

def plot_delay_duration(accident_df):
    plt.style.use('ggplot')
    # Filter outliers
    plot1_df = accident_df[accident_df['Duration'] <= 50]

    # Create duration plots by severity
    df1 = plot1_df.groupby(['Severity']).agg(mean_duration=('Duration','mean'))
    df1 = df1.reset_index()

    # Create duration plots by cities and severity
    df2 = plot1_df.groupby(['Severity', 'City']).agg(mean_duration=('Duration','mean'))
    df2 = df2.reset_index()

    # Create count plots by severity
    df3 = plot1_df.groupby(['Severity']).agg(count=('Duration','count'))
    df3 = df3.reset_index()

    # Create count plots by cities and severity
    df4 = plot1_df.groupby(['Severity', 'City']).agg(count=('Duration','count'))
    df4 = df4.reset_index()


    fig, axs = plt.subplots(2, 2, figsize=(15,10))

    fig.suptitle('Delay Duration Analysis Due to Accidents')
    sns.barplot(ax=axs[0, 0], x="Severity", y="mean_duration", data=df1)
    axs[0,0].set_title("Overall Mean Duration By Severity")
    axs[0,0].set_xlabel("Severity")
    axs[0,0].set_ylabel("Mean Delay (Hrs)")

    sns.barplot(ax=axs[0, 1], x="Severity", y="count", data=df3)
    axs[0,1].set_title("Overall Accident Count By Severity")
    axs[0,1].set_xlabel("Severity")
    axs[0,1].set_ylabel("Accident Count")


    sns.barplot(ax=axs[1, 0], x="City", y="mean_duration", hue="Severity", data=df2)
    axs[1, 0].set_title("Mean Duration by Severity in Each City")
    axs[1, 0].set_xlabel("City")
    axs[1, 0].set_ylabel("Mean Delay (Hrs)")

    sns.barplot(ax=axs[1, 1], x="City", y="count", hue="Severity", data=df4)
    axs[1, 1].set_title("Accident Count by Severity in Each City")
    axs[1, 1].set_xlabel("City")
    axs[1, 1].set_ylabel("Accident Count")

    plt.show()

def plot_mean_distance(accident_df):
    # Filter outliers
    plot1_df = accident_df[accident_df['Distance(mi)'] <= 50]

    # Create duration plots by severity
    df1 = plot1_df.groupby(['Severity']).agg(mean_distance=('Distance(mi)','mean'))
    df1 = df1.reset_index()

    # Create duration plots by cities and severity
    df2 = plot1_df.groupby(['Severity', 'City']).agg(mean_distance=('Distance(mi)','mean'))
    df2 = df2.reset_index()

    # Create count plots by severity
    df3 = plot1_df.groupby(['Severity']).agg(count=('Distance(mi)','count'))
    df3 = df3.reset_index()

    # Create count plots by cities and severity
    df4 = plot1_df.groupby(['Severity', 'City']).agg(count=('Distance(mi)','count'))
    df4 = df4.reset_index()


    fig, axs = plt.subplots(2, 2, figsize=(15,10))

    fig.suptitle('Congestion Analysis Due to Accidents')
    sns.barplot(ax=axs[0, 0], x="Severity", y="mean_distance", data=df1)
    axs[0,0].set_title("Congestion Distance By Severity")
    axs[0,0].set_xlabel("Severity")
    axs[0,0].set_ylabel("Mean Congestion Distance (mi)")

    sns.barplot(ax=axs[0, 1], x="Severity", y="count", data=df3)
    axs[0,1].set_title("Overall Accident Count By Severity")
    axs[0,1].set_xlabel("Severity")
    axs[0,1].set_ylabel("Accident Count")


    sns.barplot(ax=axs[1, 0], x="City", y="mean_distance", hue="Severity", data=df2)
    axs[1, 0].set_title("Mean Congestion Distance by Severity in Each City")
    axs[1, 0].set_xlabel("City")
    axs[1, 0].set_ylabel("Mean Congestion Distance (mi)")

    sns.barplot(ax=axs[1, 1], x="City", y="count", hue="Severity", data=df4)
    axs[1, 1].set_title("Accident Count by Severity in Each City")
    axs[1, 1].set_xlabel("City")
    axs[1, 1].set_ylabel("Accident Count")

    plt.show()

def plot_day_night(accident_df):
    # Accident Distribution by Severity and Sunrise/Sunset
    plot1_df = accident_df
    df0 = plot1_df.groupby(['Sunrise_Sunset']).agg(count=('Severity','count'))
    df0 = df0.reset_index()

    df1 = plot1_df.groupby(['Severity', 'Sunrise_Sunset']).agg(count=('Severity','count'))
    df1 = df1.reset_index()

    plot2_df = accident_df[accident_df['Severity']==1]
    df2 = plot2_df.groupby(['City', 'Sunrise_Sunset']).agg(count=('Severity','count'))
    df2 = df2.reset_index()

    plot3_df = accident_df[accident_df['Severity']==0]
    df3 = plot3_df.groupby(['City', 'Sunrise_Sunset']).agg(count=('Severity','count'))
    df3 = df3.reset_index()

    fig, axs = plt.subplots(2, 2, figsize=(15,10))

    fig.suptitle('Accident Distribution by Day and Night')
    sns.barplot(ax=axs[0, 0], x="Sunrise_Sunset", y="count", data=df0)
    axs[0,0].set_title("Overall Accidents by Day & Night")
    axs[0,0].set_xlabel("Day/Night")
    axs[0,0].set_ylabel("Accident Count")

    sns.barplot(ax=axs[0, 1], x="Sunrise_Sunset", y="count", hue="Severity", data=df1)
    axs[0,1].set_title("Accident Count By Severity and Day/Night")
    axs[0,1].set_xlabel("Day/Night")
    axs[0,1].set_ylabel("Accident Count")


    sns.barplot(ax=axs[1, 0], x="City", y="count", hue="Sunrise_Sunset", data=df2)
    axs[1, 0].set_title("Severity 1 Accident Distribution by City and Day/Night")
    axs[1, 0].set_xlabel("City")
    axs[1, 0].set_ylabel("Accident Count")

    sns.barplot(ax=axs[1, 1], x="City", y="count", hue="Sunrise_Sunset", data=df3)
    axs[1, 1].set_title("Severity 0 Accident Distribution by City and Day/Night")
    axs[1, 1].set_xlabel("City")
    axs[1, 1].set_ylabel("Accident Count")

    plt.show()

def plot_weather_conditions(accident_df):
    plot1_df = accident_df
    df0 = plot1_df.groupby(['Weather_Condition']).agg(count=('Severity','count'))
    df0 = df0.sort_values(['count'], ascending=False)[:10]
    df0 = df0.reset_index()

    plot2_df = accident_df[accident_df['Severity']==1]
    df1 = plot2_df.groupby(['Weather_Condition']).agg(count=('Severity','count'))
    df1 = df1.sort_values(['count'], ascending=False)[:10]
    df1 = df1.reset_index()

    plot3_df = accident_df[accident_df['Severity']==0]
    df2 = plot3_df.groupby(['Weather_Condition']).agg(count=('Severity','count'))
    df2 = df2.sort_values(['count'], ascending=False)[:10]
    df2 = df2.reset_index()

    # plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(18,18))

    # fig.suptitle('Accidents Distribution by Weather Condition')

    sns.barplot(ax=axs[0], x="Weather_Condition", y="count", data=df0)
    axs[0].set_xlabel("Weather Condition")
    axs[0].set_ylabel("Accident Count")
    axs[0].set_title("Overall Accident Distribution by Weather Condition")

    sns.barplot(ax=axs[1], x="Weather_Condition", y="count", data=df1)
    axs[1].set_xlabel("Weather Condition")
    axs[1].set_ylabel("Accident Count")
    axs[1].set_title("Severity 1 Accident Distribution by Weather Condition")

    sns.barplot(ax=axs[2], x="Weather_Condition", y="count", data=df2)
    axs[2].set_xlabel("Weather Condition")
    axs[2].set_ylabel("Accident Count")
    axs[2].set_title("Severity 0 Accident Distribution by Weather Condition")
    plt.show()

def plot_bpnn_results(title: str, losses: List, test_accuracy: float, accuracies: List,
                      times: List, subsample: int = 1, min_acc: float = 0.0):
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
    ax[0][0].set_ylim([min_acc, 1.0])
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
    make_space_above(ax, top_margin=2)
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
