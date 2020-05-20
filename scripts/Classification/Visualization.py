import pandas as pd
import matplotlib.pyplot as plt
import plotFunctions as func
from imblearn.over_sampling import SMOTE


class Visualization:

    def __init__(self):
        pass

    def heatmap(self):
        fig = plt.figure(figsize=[12, 12])
        axs = plt.axes()
        corr_mtx = self.data.corr()
        func.draw_heatmap(ax=axs, data=corr_mtx)

    def missing_values(self):
        fig = plt.figure(figsize=(10, 7))
        mv = {}
        data = self.data
        for var in data:
            mv[var] = data[var].isna().sum()
            func.bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var,
                           'nr. missing values')
        fig.tight_layout()
        plt.show()

    def single_boxplots(self):
        self.data.boxplot(figsize=(10, 6))
        plt.show()

    def multiple_boxplots(self):
        columns = self.data.select_dtypes(include='number').columns
        rows, cols = func.choose_grid(len(columns))
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
        i, j = 0, 0
        for n in range(len(columns)):
            axs[i, j].set_title('Boxplot for %s' % columns[n])
            axs[i, j].boxplot(data[columns[n]].dropna().values)
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        fig.tight_layout()
        plt.show()

    def multiple_hist(self):
        columns = self.data.select_dtypes(include='number').columns
        rows, cols = func.choose_grid(len(columns))
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
        i, j = 0, 0
        for n in range(len(columns)):
            axs[i, j].set_title('Histogram for %s' % columns[n])
            axs[i, j].set_xlabel(columns[n])
            axs[i, j].set_ylabel("probability")
            axs[i, j].hist(data[columns[n]].dropna().values, 'auto')
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        fig.tight_layout()
        plt.show()

    def scatter_plot(self):
        columns = self.data.select_dtypes(include='number').columns
        rows, cols = len(columns) - 1, len(columns) - 1
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
        for i in range(len(columns)):
            var1 = columns[i]
            for j in range(i + 1, len(columns)):
                var2 = columns[j]
                axs[i, j - 1].set_title("%s x %s" % (var1, var2))
                axs[i, j - 1].set_xlabel(var1)
                axs[i, j - 1].set_ylabel(var2)
                axs[i, j - 1].scatter(data[var1], data[var2])
        fig.tight_layout()
        plt.show()

    def data_balance(self):
        target_count = self.data['bug'].value_counts()

        plt.figure()
        plt.title('Class balance')
        plt.bar(target_count.index, target_count.values)

        min_class = target_count.idxmin()
        ind_min_class = target_count.index.get_loc(min_class)

        print('Minority class:', target_count[ind_min_class])
        print('Majority class:', target_count[1 - ind_min_class])
        print('Proportion:', round(target_count[ind_min_class] / target_count[1 - ind_min_class], 2), ': 1')

        RANDOM_STATE = 42
        values = {'Original': [target_count.values[ind_min_class], target_count.values[1 - ind_min_class]]}

        df_class_min = self.data[self.data['bug'] == min_class]
        df_class_max = self.data[self.data['bug'] != min_class]

        df_under = df_class_max.sample(len(df_class_min))
        values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

        df_over = df_class_min.sample(len(df_class_max), replace=True)
        values['OverSample'] = [len(df_over), target_count.values[1 - ind_min_class]]

        smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
        y = self.data.pop('bug').values
        X = self.data.values
        _, smote_y = smote.fit_sample(X, y)
        smote_target_count = pd.Series(smote_y).value_counts()
        values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1 - ind_min_class]]

        plt.figure()
        func.multiple_bar_chart(plt.gca(),
                                [target_count.index[ind_min_class], target_count.index[1 - ind_min_class]],
                                values, 'Target', 'frequency', 'Class balance')
        plt.show()
