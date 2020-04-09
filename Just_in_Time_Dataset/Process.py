from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif


class Process:

    def __init__(self):
        pass

    def minMax(self):
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(self.trnX)
        self.trnX = scaler.transform(self.trnX)
        self.tstX = scaler.transform(self.tstX)

    def standard(self):
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.trnX)
        self.trnX = scaler.transform(self.trnX)
        self.tstX = scaler.transform(self.tstX)

    def smote(self):
        sampler = SMOTE(random_state=41)
        self.trnX, self.trnY = sampler.fit_resample(self.trnX, self.trnY.ravel())

    def near_miss(self):
        from imblearn.under_sampling import NearMiss
        nr = NearMiss()
        self.trnX, self.trnY = nr.fit_sample(self.trnX, self.trnY.ravel())

    def undersample(self):
        sampler = RandomUnderSampler(sampling_strategy='majority')
        self.trnX, self.trnY = sampler.fit_sample(self.trnX, self.trnY)
        print('  shape %s', str(self.trnX.shape))

    def FS(self, y, k_value, opt: str = f_classif):
        fs = SelectKBest(opt, k_value)
        fs.fit(self.trnX, self.trnY)
        self.trnX = fs.transform(self.trnX)
        self.tstX = fs.transform(self.tstX)
        print('strategy' + ' | ' + 'nr. of components' + ' | ' + ' shape trn, tst' + ' | ' + 'pvalues')
        print(opt, k_value, self.trnX.shape, self.tstX.shape, fs.pvalues_[:5])

    def pComponentAnalysis(self, k_value):
        pca = PCA(n_components=k_value)
        pca.fit(self.trnX, self.trnY)
        self.trnX = pca.transform(self.trnX)
        self.tstX = pca.transform(self.tstX)
        print('  nr. of components ' + ' | ' + 'variance ratio ' + ' | ' + 'singular values ' + ' ')
        print(k_value, pca.explained_variance_ratio_[:5], pca.singular_values_[:5])

    def LDA(self, k_value):
        lda = LinearDiscriminantAnalysis(n_components=k_value)
        self.trnX = lda.fit(self.trnX, self.tstY).transform(self.trnX)
        self.tstX = lda.transform(self.tstX)
        print('  LDA explained variance (first two components):', lda.explained_variance_ratio_[:5])

    def autoencoder(self):
        pass