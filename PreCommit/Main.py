from Data import DataCI
from Prioritizer import NNEmbeddings
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules

from RNN import RNN


def study_start_date(dates):
    acc_train = []
    acc_val = []
    loss_train = []
    loss_val = []

    for k, v in dates.items():
        commits = pd.read_csv('../pub_data/test_commits_pub.csv', encoding='latin-1', sep='\t')
        test_details = pd.read_csv('../pub_data/test_details_pub.csv', sep='\t')
        test_status = pd.read_csv('../pub_data/test_histo_pub.csv', sep='\t')
        mod_files = pd.read_csv("../pub_data/test_commits_mod_files_pub.csv", sep='\t')

        D = DataCI(commits, test_details, test_status, mod_files, start_date=v)
        a_train, l_train, a_val, l_val = NNEmbeddings(D).crossValidation(nb_epochs=5, k_folds=5, n_positive=1000)

        acc_train.append(a_train)
        acc_val.append(a_val)
        loss_train.append(l_train)
        loss_val.append(l_val)

    d = {
        'Accuracy_train': acc_train,
        'Accuracy_val': acc_val,
        'Loss_train': loss_train,
        'Loss_val': loss_val
    }

    df = pd.DataFrame(d, index=dates.keys())
    ax = df.plot.bar(rot=0)
    ax.set_xlabel('Months')
    ax.set_ylabel('Percentage')
    ax.set_title('Removal of Files that have not been modified for some time')
    plt.show()


def arm(df, fgrowth=False):
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(sparse_output=True)
    df = df[['mod_files', 'name']]

    df = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(df.pop('mod_files')),
            index=df.index,
            columns=mlb.classes_))

    df = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(df.pop('name')),
            index=df.index,
            columns=mlb.classes_))
    print(df)

    if fgrowth:
        frequent_itemsets = fpgrowth(df, min_support=0.02, use_colnames=True)
    else:
        frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
    print(frequent_itemsets)

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
    print(rules.head())


def main():
    commits = pd.read_csv('../pub_data/test_commits_pub.csv', encoding='latin-1', sep='\t')
    test_details = pd.read_csv('../pub_data/test_details_pub.csv', sep='\t')
    test_status = pd.read_csv('../pub_data/test_histo_pub.csv', sep='\t')
    mod_files = pd.read_csv("../pub_data/test_commits_mod_files_pub.csv", sep='\t')

    dates = {'3': '2019-12-11 00:00:00.000000',
             '6': '2019-09-11 00:00:00.000000',
             '12': '2019-03-11 00:00:00.000000',
             '18': '2018-09-11 00:00:00.000000',
             '24': '2018-03-11 00:00:00.000000',
             '30': '2017-09-11 00:00:00.000000',
             '36': '2017-03-11 00:00:00.000000',
             '42': '2016-09-11 00:00:00.000000',
             '48': '2016-03-11 00:00:00.000000',
             }

    # study_start_date(dates)

    D = DataCI(commits, test_details, test_status, mod_files)
    arm(D.df_link)

    # rnn = RNN(D)
    # Prio = NNEmbeddings(D)

    # TSNE Plots
    # Prio.plot_embeddings()
    # Prio.plot_embeddings_labeled(layer='tests')
    # Prio.plot_embeddings_labeled(layer='files')

    # UMAP Plots
    # Prio.plot_embeddings(method='UMAP')
    # Prio.plot_embeddings_labeled(layer='tests', method='UMAP')
    # Prio.plot_embeddings_labeled(layer='files', method='UMAP')


if __name__ == '__main__':
    main()
