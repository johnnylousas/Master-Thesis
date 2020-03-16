from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif


def minMax(trn_x, tst_x):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(trn_x)
    trn_x = scaler.transform(trn_x)
    tst_x = scaler.transform(tst_x)
    return trn_x, tst_x


def standard(trn_x, tst_x):
    scaler = preprocessing.StandardScaler()
    scaler.fit(trn_x)
    trn_x = scaler.transform(trn_x)
    tst_x = scaler.transform(tst_x)

    return trn_x, tst_x


def smote(trn_x, trn_y):
    sampler = SMOTE(random_state=41)
    trn_x, trn_y = sampler.fit_resample(trn_x, trn_y.ravel())
    return trn_x, trn_y


def near_miss(trn_x, trn_y):
    from imblearn.under_sampling import NearMiss
    nr = NearMiss()
    trn_x, trn_y = nr.fit_sample(trn_x, trn_y.ravel())
    return trn_x, trn_y


def undersample(trn_x, trn_y):
    sampler = RandomUnderSampler(sampling_strategy='all', ratio={1: 200})
    trn_x, trn_y = sampler.fit_sample(trn_x, trn_y)
    print('  shape %s', str(trn_x.shape))
    return trn_x, trn_y


def FS(trn_x, tst_x, y, k_value, opt: str = f_classif):
    fs = SelectKBest(opt, k_value)
    fs.fit(trn_x, y)
    trn_x = fs.transform(trn_x)
    tst_x = fs.transform(tst_x)
    print('strategy' + ' | ' + 'nr. of components' + ' | ' + ' shape trn, tst' + ' | ' + 'pvalues')
    print(opt, k_value, trn_x.shape, tst_x.shape, fs.pvalues_[:5])

    return trn_x, tst_x


def PrincipalComponentAnalysis(trn_x, tst_x, trn_y, k_value):
    pca = PCA(n_components=k_value)
    pca.fit(trn_x, trn_y)
    trn_x = pca.transform(trn_x)
    tst_x = pca.transform(tst_x)
    print('  nr. of components ' + ' | ' + 'variance ratio ' + ' | ' + 'singular values ' + ' ')
    print(k_value, pca.explained_variance_ratio_[:5], pca.singular_values_[:5])
    return trn_x, tst_x


def LDA(trn_x, tst_x, y, k_value):
    lda = LinearDiscriminantAnalysis(n_components=k_value)
    trn_x = lda.fit(trn_x, y).transform(trn_x)
    tst_x = lda.transform(tst_x)
    print('  LDA explained variance (first two components):', lda.explained_variance_ratio_[:5])
    return trn_x, tst_x
