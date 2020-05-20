import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import numpy as np


class LayoutStyleObject:
    # defined ro describe graph layout style
    def __init__(self, title: str, xlabel: str, ylabel: str,
                 xmin: int = 1, xmax: int = 1, ymin: int = 1, ymax: int = 1,
                 grid: bool = False, grid_color: str = 'grey', grid_linestyle: str = '--', grid_linewidth: float = 0.5):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.grid = grid
        self.grid_color = grid_color
        self.grid_linestyle = grid_linestyle
        self.grid_linewidth = grid_linewidth


class PlotStyleObject:
    # defined to describe plot style
    def __init__(self, color: str, legend: str = ' ', marker: str = ' ', markersize: float = 5,
                 linewidth: float = 1, linestyle: str = '-', alpha: float = 1, mfc: str = ' '):
        self.color = color
        self.legend = legend
        self.marker = marker
        self.markersize = markersize
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.alpha = alpha
        if mfc == ' ':
            self.mfc = color  # color to fill with
        else:
            self.mfc = mfc


# plot many 2D plots using StyleObject --------------------------------
def multiple_plots(ax: plt.Axes, xvalues: dict, yvalues: dict, layout: LayoutStyleObject, plotstyle: dict,
                   legends: bool = False):
    legend: list = []
    ax.set_title(layout.title)
    ax.set_xlabel(layout.xlabel)
    ax.set_ylabel(layout.ylabel)
    # check if x scale is changed
    if layout.xmin == layout.xmax:
        ax.autoscale(True, 'x')
    else:
        ax.set_xlim(layout.xmin, layout.xmax)
    # check if y scale is changed
    if layout.ymin == layout.ymax:
        ax.autoscale(True, 'y')
    else:
        ax.set_ylim(layout.ymin, layout.ymax)
    # check grid configurations
    if layout.grid:
        plt.grid(color=layout.grid_color, linestyle=layout.grid_linestyle, linewidth=layout.grid_linewidth)
    # draw plots on figure
    for name, y in yvalues.items():
        aux = plotstyle.get(name)
        plt.plot(xvalues.get(name), y, color=aux.color, marker=aux.marker, markersize=aux.markersize,
                 linewidth=aux.linewidth, linestyle=aux.linestyle, alpha=aux.alpha, mfc=aux.mfc)
        legend.append(aux.legend)
    # set legends if request
    if legends:
        ax.legend(legend, loc='best', fancybox=True, shadow=True)


def joint_plot(var_x, var_y, color, kind='scatter', space: int = 0, plot_joint: bool = False, grid: bool = False):
    sns.set()
    # Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
    if plot_joint:
        g1 = (sns.jointplot(x=var_x, y=var_y, kind=kind, space=space, color=color)
              .plot_joint(sns.kdeplot, zorder=0, n_levels=3))
    else:
        sns.jointplot(x=var_x, y=var_y, kind=kind, space=space, color=color)
    if grid:
        plt.grid(True, linewidth=0.5)
    plt.show()


def csv_to_data_frame(file: str, index_col: str, skiprows: int = 0,
                      parse_dates: bool = True, infer_datetime_format: bool = True):
    register_matplotlib_converters()
    return pd.read_csv(file, skiprows=skiprows, index_col=index_col, parse_dates=parse_dates,
                       infer_datetime_format=infer_datetime_format)


# --------------------------------------------------- various plots


def choose_grid(nr):
    return nr // 4 + 1, 4


def line_chart(ax: plt.Axes, series: pd.Series, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(series)


def scatter(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    plt.scatter(x=xvalues, y=yvalues)


def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                        percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox=True, shadow=True)


def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=90, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor='grey')


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                       percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name)
        k += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True)


# --------------------------------------------------- heatmap plots

def draw_heatmap(ax: plt.Axes, data, xticklabels='auto', yticklabels='auto', title: str = '', annot: bool = False,
                 cmap: str = 'Blues'):
    ax = sns.heatmap(data=data, xticklabels=xticklabels, yticklabels=yticklabels, linewidth=.5, annot=annot,
                     cmap=cmap, annot_kws={"size": 7})
    ax.set_ylim(len(data.index), 0)
    plt.yticks(rotation=0)
    if title != '':
        ax.set_title(title)
    plt.show()


def plot_confusion_matrix(ax: plt.Axes, data, labels, xticklabels='auto', yticklabels='auto', title: str = '',
                          annot: bool = False, cmap: str = 'seismic'):
    ax = sns.heatmap(data, annot=True, ax=ax)  # annot=True to annotate cells
    ax.set_ylim(len(data), 0)
    plt.yticks(rotation=0)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.show()


# ------------------- functions for histogram fit gaussian and exponential


def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)' % (mean, sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
    #  sigma, loc, scale = _stats.lognorm.fit(x_values)
    #  distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)' % (1 / scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
    # a, loc, scale = _stats.skewnorm.fit(x_values)
    # distributions['SkewNorm(%.2f)'%a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions


def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 80, density=True, edgecolor='grey')
    distributions = compute_known_distributions(values, bins)
    multiple_line_chart(ax, values, distributions, 'Best fit for %s' % var, var, 'probability')


# testing -------------------------------------------------------------


""" Testing
# Figure size x[cm] by y[cm]
plt.figure(figsize=(7, 5))
# define layout style object
layout1 = LayoutStyleObject('title', 'xlabel', 'ylabel', True)
# define first plot style object
plot1 = {'one': PlotStyleObject(color='gray', legend='One', marker='o'),
         'two': PlotStyleObject(color='blue', marker='s', markersize=10, linewidth=2, linestyle='--'),
         'three': PlotStyleObject('orange')}
# build graph with style objects
plot_multiple_plots(ax=plt.gca(),
                    xvalues={'one': [0,1,2], 'two': [0,0.5,2.5], 'three': [0,1.5,3]},
                    yvalues={'one': [2,4,7], 'two': [3,7,10], 'three': [7,3,1]},
                    layout=layout1, plotstyle=plot1)
# show graph
plt.show()
"""
