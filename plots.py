import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_error_vs_time(df, 
                       color_param='simulator', 
                       marker_param='rank', 
                       hue_order=None, 
                       style_order=None):

    df_copy = df.copy()

    plt.figure(figsize=(14, 8))

    sns.lineplot(
        data=df_copy,
        x='time',
        y='error',
        hue=color_param,
        style=marker_param,
        palette='deep',
        markers=True,
        alpha=0.7,
        hue_order=hue_order,
        style_order=style_order
    )
    
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Error (L2 norm)', fontsize=14)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid()
    plt.show()