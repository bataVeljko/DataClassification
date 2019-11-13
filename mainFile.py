import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
import time
from joblib import dump, load
import os

""" fja koja vraca DataFrame bez 0 redova ili 0 kolona"""
def remove_zero(df):
    izbaceni = df[df.sum(axis=1) == 0]
    df.drop(df[df.sum(axis=1) == 0].index, inplace=True)
    return df, izbaceni

""" Otkrivanje i izdvajanje elemenata van granica"""
def detect_and_exclude_outliers(df):
    outliers_small = df[(stats.zscore(df) <= -3).any(axis=1)]
    outliers_big = df[(stats.zscore(df) >= 3).any(axis=1)]
    df_without_outliers = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df_without_outliers, outliers_big, outliers_small

def fit_model(model, x_train, x_test, y_train, y_test, mode, method):
    if mode == 'save':
        model.fit(x_train, y_train.ravel())
        dump(model, os.path.join('models', method + '.joblib'))
    print(f"Rezultat trening skupa: {model.score(x_train, y_train):.3f}")
    print(f"Rezultat test skupa: {model.score(x_test, y_test):.3f}")

def prediction(model, x_train, x_test, y_train, y_test):
    y_train_predicted = model.predict(x_train)
    y_test_predicted = model.predict(x_test)
    print("Matrica kofuzije trening skupa:\n" + str(confusion_matrix(y_train, y_train_predicted)))
    print("Matrica kofuzije test skupa:\n" + str(confusion_matrix(y_test, y_test_predicted)))

# Klasifikacija pomocu K najblizih suseda, rtype predstavlja tip podatka koji cemo klasifikovati
# 'gen' za klasifikaciju gena i 'out' za klasifikaciju elemenata van granice
def knn(x_train, x_test, y_train, y_test, non, mode, rtype):
    print("-----K najblizih suseda:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'knn_' + rtype + '.joblib'))
    else:
        model = KNeighborsClassifier(n_neighbors=non, weights='uniform')
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'knn_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Decision tree classifier
def dtc(x_train, x_test, y_train, y_test, mode, rtype, criteria, depth):
    print("-----Drvo odlucivanja:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'dtc_' + rtype + '.joblib'))
    else:
        model = DecisionTreeClassifier(criterion=criteria, max_depth=depth)
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'dtc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Support vector machine
def svm(x_train, x_test, y_train, y_test, kernel, mode, rtype):
    print("-----SVM:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'svm_' + rtype + '.joblib'))
    else:
        model = SVC(kernel=kernel, degree=2, gamma='scale')
    # noinspection PyTypeChecker
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'svc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Random forest classifier
def rfc(x_train, x_test, y_train, y_test, n_est, mode, rtype):
    print("-----Nasumicna suma:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'rfc_' + rtype + '.joblib'))
    else:
        model = RandomForestClassifier(n_estimators=n_est, max_depth=5, criterion='gini')
    # noinspection PyTypeChecker
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'rfc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Bagging
def bag(x_train, x_test, y_train, y_test, n_est, mode, rtype, model):
    print("-----Bagging:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'bagging_' + rtype + '.joblib'))
    else:
        model = BaggingClassifier(base_estimator=model, n_estimators=n_est)
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'bagging_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Boosting
def boost(x_train, x_test, y_train, y_test, n_est, mode, rtype):
    print("-----Boosting:-----")
    start = time.time()
    model = DecisionTreeClassifier(max_depth=5)
    if mode == 'load':
        model = load(os.path.join('models', 'boosting_' + rtype + '.joblib'))
    else:
        model = AdaBoostClassifier(base_estimator=model, n_estimators=n_est)
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'boosting_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


# Voting classifier
def vot(x_train, x_test, y_train, y_test, mode, rtype):
    print("-----Voting:-----")
    start = time.time()
    est_model1 = RandomForestClassifier(n_estimators=100, max_depth=5)
    est_model2 = SVC(kernel='linear', gamma='scale')
    est_model3 = DecisionTreeClassifier(max_depth=5)
    if mode == 'load':
        model = load(os.path.join('models', 'voting_' + rtype + '.joblib'))
    else:
        model = VotingClassifier(estimators=[('dtc', est_model1), ('svc', est_model2),
                                             ('tree', est_model3)])

    est_model1.fit(x_train, y_train.ravel())
    est_model2.fit(x_train, y_train.ravel())
    est_model3.fit(x_train, y_train.ravel())

    fit_model(model, x_train, x_test, y_train, y_test, mode, 'voting_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print(f"Vreme izvrsavanja: {time.time()-start:.3f} \n")


def main():
    # Ucitavanje podataka
    first_file = '029_pluripotent_stem_cell-derived_SCGB3A2+_airway_epithelium_csv.csv'
    second_file = '030_pluripotent_stem_cell-derived_SCGB3A2+_airway_epithelium_csv.csv'
    df_first = pd.read_csv(first_file).set_index('geni')
    df_second = pd.read_csv(second_file).set_index('geni')

    """
    ************************
        Pretprocesiranje
    ************************
    """

    # dfft -> data frame first transposed
    # dfst -> data frame second transposed
    # dfft i dfst sadrze transponovane podatke iz tablea df_first i df_second
    dfft = df_first.T
    dfst = df_second.T

    # Dodavanje kolone na kraju kao oznaka klase
    dfft['class'] = 1
    dfst['class'] = 2

    print("Dimenzija podataka prve datoteke: " + str(dfft.shape))
    print("Dimenzija podataka druge datoteke: " + str(dfst.shape) + '\n')

    # Spajanje podataka u jednu matricu, koju cemo nadalje koristiti u toku pretprocesiranja,
    # a zatim i klasifikacije
    dft = pd.concat([dfft, dfst])

    print("Dimenzija spojene tabele: " + str(dft.shape) + "\n")

    # Provera da li skup podataka sadrzi neku NaN vrednost
    if dft.isnull().values.any():
        print("Skup sadrzi NaN vrednosti!\n")
    else:
        print("Skup ne sadrzi nijednu NaN vrednost\n")

    # Uklanjanje nula-redova
    dft, zero_rows = remove_zero(dft)
    print("Dimenzija podataka nakon uklanjanja 0 redova: " + str(dft.shape))

    # Uklanjanje nula-kolona
    df, zero_cols = remove_zero(dft.T)
    dft = df.T
    print("Dimenzija podataka nakon uklanjanja 0 kolona: " + str(dft.shape))
    print("Broj 0 kolona: " + str(zero_cols.shape[0]))

    # Izdvajanje elemenata van granica
    df, outliers_big, outliers_small = detect_and_exclude_outliers(dft.T)
    dft = df.T
    outliers_small = outliers_small.T
    outliers_big = outliers_big.T

    print("Dimenzija podataka nakon uklanjanja elemenata van granica: " + str(dft.shape))
    print("Dimenzija elemenata van donje granice: " + str(outliers_small.shape))
    print("Dimenzija elemenata van gornje granice: " + str(outliers_big.shape) + "\n")

    """ Posto smo u toku izdvajanja elemnata van granica izgubili podatke o klasi kojoj elementi van granica pripadaju
        moramo ponovo navesti kako bismo mogli pravilno izvrsiti klasifikaciju kasnije, a kako smo prethodnim ispisom
        utvrdili da ne postoje elementi van donje granice, mozemo originalnu kolekciju koristiti u nastavku"""
    class_column = dft.values[:, -1:]
    outliers = outliers_big
    outliers['class'] = class_column

    # print("Deo podataka za klasifikaciju:\n" + str(dft) + '\n')
    # print("Deo elemenata van granica:\n" + str(outliers) + '\n')

    """
    ************************
        Klasifikacija
    ************************
    """

    # X predstavlja podatke za klasifikaciju
    x = dft.values[:, :-1]

    # ox predstavlja elemente van granica za klasifikaciju
    ox = outliers.values[:, :-1]

    # Y predstavlja klase kojima podaci pripadaju
    y = dft.values[:, -1:]

    # oy predstavlja klase kojima elementi van granica pripadaju, medjutim kako je taj podatak jednak koloni y,
    # samo cemo izjednaciti ove dve vrednosti
    oy = y

    # print(x)
    # print(y)
    # print(ox)
    # print(oy)

    # Podela podataka na trening i test skup, odnos 70/30
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    ox_train, ox_test, oy_train, oy_test = train_test_split(ox, oy, test_size=0.3)

    # ***** Jednostavne metode *****

    # K najblizih suseda
    """
    knn(x_train, x_test, y_train, y_test, non=3, mode='save', rtype='gen')
    knn(x_train, x_test, y_train, y_test, non=5, mode='save', rtype='gen')
    knn(x_train, x_test, y_train, y_test, non=10, mode='save', rtype='gen')
    knn(ox_train, ox_test, oy_train, oy_test, non=3, mode='save', rtype='out')
    knn(ox_train, ox_test, oy_train, oy_test, non=5, mode='save', rtype='out')
    knn(ox_train, ox_test, oy_train, oy_test, non=10, mode='save', rtype='out')
    """

    # Drvo odlucivanja
    """
    dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen', criteria='gini', depth=None)
    dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen', criteria='entropy', depth=None)
    dtc(ox_train, ox_test, oy_train, oy_test, mode='save', rtype='out', criteria='gini', depth=None)
    dtc(ox_train, ox_test, oy_train, oy_test, mode='save', rtype='out', criteria='entropy', depth=None)
    dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen', criteria='gini', depth=5)
    dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen', criteria='entropy', depth=5)
    dtc(ox_train, ox_test, oy_train, oy_test, mode='save', rtype='out', criteria='gini', depth=5)
    dtc(ox_train, ox_test, oy_train, oy_test, mode='save', rtype='out', criteria='entropy', depth=5)
    """

    # Masine sa potpornim vektorima
    """
    svm(x_train, x_test, y_train, y_test, kernel='rbf', mode='save', rtype='gen')
    svm(x_train, x_test, y_train, y_test, kernel='linear', mode='save', rtype='gen')
    svm(x_train, x_test, y_train, y_test, kernel='poly', mode='save', rtype='gen')
    svm(ox_train, ox_test, oy_train, oy_test, kernel='rbf', mode='save', rtype='out')
    svm(ox_train, ox_test, oy_train, oy_test, kernel='linear', mode='save', rtype='out')
    svm(ox_train, ox_test, oy_train, oy_test, kernel='poly', mode='save', rtype='out')
    """

    # ***** Ansambl tehnike *****

    # Nasumicna suma
    """
    rfc(x_train, x_test, y_train, y_test, n_est=10, mode='save', rtype='gen')
    rfc(x_train, x_test, y_train, y_test, n_est=50, mode='save', rtype='gen')
    rfc(x_train, x_test, y_train, y_test, n_est=100, mode='save', rtype='gen')
    rfc(ox_train, ox_test, oy_train, oy_test, n_est=10, mode='save', rtype='out')
    rfc(ox_train, ox_test, oy_train, oy_test, n_est=50, mode='save', rtype='out')
    rfc(ox_train, ox_test, oy_train, oy_test, n_est=100, mode='save', rtype='out')
    """

    # Pakovanje
    """
    bag(x_train, x_test, y_train, y_test, n_est=10, mode='save', rtype='gen',
       model = DecisionTreeClassifier(max_depth=5))
    bag(x_train, x_test, y_train, y_test, n_est=50, mode='save', rtype='gen',
       model = DecisionTreeClassifier(max_depth=5))
    bag(x_train, x_test, y_train, y_test, n_est=5, mode='save', rtype='gen',
       model = SVC(kernel='linear', gamma='scale'))
    bag(x_train, x_test, y_train, y_test, n_est=20, mode='save', rtype='gen',
       model = SVC(kernel='linear', gamma='scale'))
    bag(ox_train, ox_test, oy_train, oy_test, n_est=10, mode='save', rtype='out',
       model = DecisionTreeClassifier(max_depth=5))
    bag(ox_train, ox_test, oy_train, oy_test, n_est=50, mode='save', rtype='out',
       model = DecisionTreeClassifier(max_depth=5))
    bag(ox_train, ox_test, oy_train, oy_test, n_est=5, mode='save', rtype='out',
       model = SVC(kernel='linear', gamma='scale'))
    bag(ox_train, ox_test, oy_train, oy_test, n_est=20, mode='save', rtype='out',
       model = SVC(kernel='linear', gamma='scale'))
    """

    # Pojacavanje
    """
    boost(x_train, x_test, y_train, y_test, n_est=10, mode='save', rtype='gen')
    boost(x_train, x_test, y_train, y_test, n_est=50, mode='save', rtype='gen')
    boost(x_train, x_test, y_train, y_test, n_est=100, mode='save', rtype='gen')
    boost(ox_train, ox_test, oy_train, oy_test, n_est=10, mode='save', rtype='out')
    boost(ox_train, ox_test, oy_train, oy_test, n_est=50, mode='save', rtype='out')
    boost(ox_train, ox_test, oy_train, oy_test, n_est=100, mode='save', rtype='out')
    """

    # Glasanje
    """
    vot(x_train, x_test, y_train, y_test, mode='save', rtype='gen')
    vot(ox_train, ox_test, oy_train, oy_test, mode='save', rtype='out')
    """

if __name__ == '__main__':
    main()