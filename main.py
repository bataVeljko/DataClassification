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

""" fja koja vraca DataFrame bez 0 redova
    eksperimentalnim putem je utvrdjeno da u oba fajla ne postoje 0 kolone, pa one nisu ispitivane """
def remove_zero_rows(df):
    izbaceni = df[df.sum(axis=1) == 0]
    df.drop(df[df.sum(axis=1) == 0].index, inplace=True)
    return df, izbaceni


""" Otkrivanje i izdvajanje elemenata van granica
    Ispitivanjem podataka, dosao sam do zakljucka da elemente van granica predstavljaju redovi, 
    koji u jednoj ili vise kolona ima znacajno vecu vrednost nego ostali clanovi"""
def detect_and_exclude_outliers(df):
    ouliers = df[(np.abs(stats.zscore(df)) >= 3).any(axis=1)]
    df_without_outliers = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return ouliers, df_without_outliers


def fit_model(model, x_train, x_test, y_train, y_test, mode, method):
    if mode == 'save':
        model.fit(x_train, y_train.ravel())
        dump(model, os.path.join('models', method + '.joblib'))
    print("Rezultat trening skupa: " + str(model.score(x_train, y_train)))
    print("Rezultat test skupa: " + str(model.score(x_test, y_test)))


def prediction(model, x_train, x_test, y_train, y_test):
    y_train_predicted = model.predict(x_train)
    y_test_predicted = model.predict(x_test)
    print("Matrica kofuzije trening skupa:\n" + str(confusion_matrix(y_train, y_train_predicted)))
    print("Matrica kofuzije test skupa:\n" + str(confusion_matrix(y_test, y_test_predicted)))


# K neighbors classifier, rtype represent type of row that are going to be classified, 'gen' or 'cell'
def knn(x_train, x_test, y_train, y_test, non, mode, rtype):
    print("-----K najblizih suseda:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'knn_' + rtype + '.joblib'))
    else:
        model = KNeighborsClassifier(n_neighbors=non, weights='uniform')
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'knn_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print("Vreme izvrsavanja: " + str(time.time()-start) + '\n')


# Decision tree classifier
def dtc(x_train, x_test, y_train, y_test, mode, rtype):
    print("-----Drvo odlucivanja:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'dtc_' + rtype + '.joblib'))
    else:
        model = DecisionTreeClassifier(criterion='entropy')
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'dtc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print("Vreme izvrsavanja: " + str(time.time()-start) + '\n')


# Support vector machine
def svm(x_train, x_test, y_train, y_test, c, kernel, gamma, mode, rtype):
    print("-----SVM:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'svm_' + rtype + '.joblib'))
    else:
        model = SVC(C=c, kernel=kernel, gamma=gamma, degree=2)
    # noinspection PyTypeChecker
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'svc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print("Vreme izvrsavanja: " + str(time.time()-start) + '\n')


# Random forest classifier
def rfc(x_train, x_test, y_train, y_test, n_est, mode, rtype):
    print("-----Nasumicna suma:-----")
    start = time.time()
    if mode == 'load':
        model = load(os.path.join('models', 'rfc_' + rtype + '.joblib'))
    else:
        model = RandomForestClassifier(n_estimators=n_est)
    # noinspection PyTypeChecker
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'rfc_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print("Vreme izvrsavanja: " + str(time.time()-start) + '\n')


# Bagging
def bag(x_train, x_test, y_train, y_test, n_est, mode, rtype):
    print("-----Bagging:-----")
    start = time.time()
    est_model = DecisionTreeClassifier()
    if mode == 'load':
        model = load(os.path.join('models', 'bagging_' + rtype + '.joblib'))
    else:
        model = BaggingClassifier(base_estimator=est_model, n_estimators=n_est)
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'bagging_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print("Vreme izvrsavanja: " + str(time.time()-start) + '\n')


# Boosting
def boost(x_train, x_test, y_train, y_test, n_est, mode, rtype):
    print("-----Boosting:-----")
    start = time.time()
    est_model = DecisionTreeClassifier()
    if mode == 'load':
        model = load(os.path.join('models', 'boosting_' + rtype + '.joblib'))
    else:
        model = AdaBoostClassifier(base_estimator=est_model, n_estimators=n_est)
    fit_model(model, x_train, x_test, y_train, y_test, mode, 'boosting_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print("Vreme izvrsavanja: " + str(time.time()-start) + '\n')


# Voting classifier
def vot(x_train, x_test, y_train, y_test, power, mode, rtype):
    print("-----Voting:-----")
    start = time.time()
    est_model1 = RandomForestClassifier(n_estimators=50)
    est_model2 = SVC(C=100, kernel='linear')
    est_model3 = DecisionTreeClassifier(criterion='entropy')
    if mode == 'load':
        model = load(os.path.join('models', 'voting_' + rtype + '.joblib'))
    else:
        model = VotingClassifier(estimators=[('dtc', est_model1), ('svc', est_model2), ('tree', est_model3)],
                                 voting=power)

    est_model1.fit(x_train, y_train.ravel())
    est_model2.fit(x_train, y_train.ravel())
    est_model3.fit(x_train, y_train.ravel())

    fit_model(model, x_train, x_test, y_train, y_test, mode, 'voting_' + rtype)
    prediction(model, x_train, x_test, y_train, y_test)
    print("Vreme izvrsavanja: " + str(time.time()-start) + '\n')


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

    # Provera da skupovi podataka ne sadrze neku NaN vrednost
    # print(df_first.isnull().values.any())
    # print(df_second.isnull().values.any())

    # dimenzija DataFrame-a nakon ucitavanja podataka
    print("Dimenzija podataka prvog fajla: " + str(df_first.shape))
    print("Dimenzija podataka drugog fajla: " + str(df_second.shape) + '\n')

    # print("Mali deo podataka prvog fajla:\n" + str(df_first.head(20)))
    # print("Mali deo podataka drugog fajla:\n" + str(df_second.head(20)) + '\n')

    df_first, zero_rows_first = remove_zero_rows(df_first)
    df_second, zero_rows_second = remove_zero_rows(df_second)
    print("Dimenzija podataka prvog fajla nakon uklanjanja 0 redova i kolona: " + str(df_first.shape))
    print("Dimenzija podataka drugog fajla nakon uklanjanja 0 redova i kolona: " + str(df_second.shape) + '\n')
    # print(df_first)
    # print(df_second)

    # Uklanjanje gena iz oba fajla po kojima se podaci trenutno razlikuju
    df_first.drop(zero_rows_second.index.difference(zero_rows_first.index).values, inplace=True)
    df_second.drop(zero_rows_first.index.difference(zero_rows_second.index).values, inplace=True)
    print("Dimenzija podataka prvog fajla nakon uklanjanja 0 redova drugog fajla: " + str(df_first.shape))
    print("Dimenzija podataka drugog fajla nakon uklanjanja 0 redova prvog fajla: " + str(df_second.shape) + '\n')

    outliers_first, df_first = detect_and_exclude_outliers(df_first)
    outliers_second, df_second = detect_and_exclude_outliers(df_second)

    print("Dimenzija podataka prvog fajla nakon uklanjanja elemenata van granica: " + str(df_first.shape))
    print("Dimenzija podataka drugog fajla nakon uklanjanja elemenata van granica: " + str(df_second.shape) + '\n')
    # print(outliers_first.head(10))
    # print(outliers_second.head(10))

    # print(outliers_first.index.intersection(outliers_second.index))

    # Uklanjanje gena iz oba fajla po kojima se podaci trenutno razlikuju
    df_first.drop(outliers_second.index.difference(outliers_first.index).values, inplace=True)
    df_second.drop(outliers_first.index.difference(outliers_second.index).values, inplace=True)
    print("Dimenzija podataka prvog fajla nakon uklanjanja elemenata van granica drugog fajla: " + str(df_first.shape))
    print("Dimenzija podataka drugog fajla nakon uklanjanja elemenata van granica prvog fajla: " + str(df_second.shape)
          + '\n')

    # Uklanjanje kolona iz prvog fajla koje ne postoje u drugom fajlu
    df_first.drop([str(i) for i in range(715, 743)], axis=1, inplace=True)

    # dfft i dfst sadrze transponovane podatke iz tablea df_first i df_second
    dfft = df_first.T
    dfst = df_second.T

    # Dodavanje kolone na kraju kao oznaka klase
    df_first['class'] = 1
    df_second['class'] = 2
    dfft['class'] = 1
    dfst['class'] = 2

    print("Dimenzija podataka prvog fajla nakon predprocesiranja: " + str(df_first.shape))
    print("Dimenzija podataka drugog fajla nakon predprocesiranja: " + str(df_second.shape) + '\n')

    # Spajanje podataka u jednu matricu
    df = pd.concat([df_first, df_second])
    dft = pd.concat([dfft, dfst])
    # df i dft sada sadrze obradjene elemente oba fajla, tako da cemo njih koristiti nadalje za klasifikaciju
    # u df redovi predstavljaju gene, a u dft-u celije
    print("Dimenzija podataka za klasifikaciju po genima" + str(df.shape))
    print("Dimenzija podataka za klasifikaciju po celijama" + str(dft.shape) + '\n')

    """
    ************************
        Klasifikacija
    ************************
    """

    # U celom delu vezanom za klasifikaciju, prvo cemo podatke klasifikovati po genima, a zatim i po brojevima celija

    # X predstavlja podatke za klasifikaciju
    x = df.values[:, :-1]
    xt = dft.values[:, :-1]

    # Y predstavlja klase kojima podaci pripadaju
    y = df.values[:, -1:]
    yt = dft.values[:, -1:]

    # print(x)
    # print(y)
    # print(xt)
    # print(yt)

    # Podela podataka na trening i test skup, odnos 70/30
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    xt_train, xt_test, yt_train, yt_test = train_test_split(xt, yt, test_size=0.3)

    # knn(x_train, x_test, y_train, y_test, non=3, mode='save', rtype='gen')
    # Rezultat trening skupa: 0.9017612361884191
    # Rezultate test skupa: 0.8427989920527234
    # knn(xt_train, xt_test, yt_train, yt_test, non=3, mode='save', rtype='cell')
    # Rezultat trening skupa: 0.977977977977978
    # Rezultat test skupa: 0.9813519813519813
    # broj suseda: 3, 5 i 10; 3 pokazalo najbolje rezultate

    # dtc(x_train, x_test, y_train, y_test, mode='save', rtype='gen')
    # Rezultat trening skupa: 0.9926476696851375
    # Rezultat test skupa: 0.8614072494669509
    # dtc(xt_train, xt_test, yt_train, yt_test, mode='save', rtype='cell')
    # Rezultat trening skupa: 1.0
    # Rezultat test skupa: 0.972027972027972

    # svm(x_train, x_test, y_train, y_test, c=100, kernel='linear', gamma=10, mode='save', rtype='gen')
    # rbf:
    # Rezultat trening skupa: 0.9926476696851375
    # Rezultat test skupa: 0.4989339019189765
    # ---> Preprilagodjavanje
    # linear:
    # Rezultat trening skupa: 0.9504860014953892
    # Rezultat test skupa: 0.9265361504167474
    # svm(xt_train, xt_test, yt_train, yt_test, c=100, kernel='linear', gamma=10, mode='save', rtype='cell')
    # rbf:
    # Rezultat trening skupa: 1.0
    # Rezultat test skupa: 0.48484848484848486
    # ---> Opet preprilagodjavanje
    # poly:
    # Rezultat trening skupa: 1.0
    # Rezultat test skupa: 0.9953379953379954
    # linear
    # Rezultat trening skupa: 1.0
    # Rezultat test skupa: 0.9976689976689976

    # Ostaje najbolji rezultat u odnosu na vreme sa drvetom odlucivanja, pa cemo njega koristiti u nastavku

    """ 
        Ansambl metode
    """

    # rfc(x_train, x_test, y_train, y_test, n_est=50, mode='save', rtype='gen')
    # Rezultat trening skupa: 0.99243997673839
    # Rezultat test skupa: 0.9163597596433417
    # rfc(xt_train, xt_test, yt_train, yt_test, n_est=50, mode='save', rtype='cell')
    # Rezultat trening skupa: 1.0
    # Rezultat test skupa: 0.9906759906759907
    # Vreme izvrsavanja: 1.1130309104919434
    # WoW

    # bag(x_train, x_test, y_train, y_test, n_est=20, mode='save', rtype='gen')
    # Rezultat trening skupa: 0.9901138157348176
    # Rezultat test skupa: 0.8988176003101376
    # bag(xt_train, xt_test, yt_train, yt_test, n_est=20, mode='save', rtype='cell')
    # Rezultat trening skupa: 1.0
    # Rezultat test skupa: 0.986013986013986

    # boost(x_train, x_test, y_train, y_test, n_est=50, mode='save', rtype='gen')
    # Rezultat trening skupa: 0.9901553543241671
    # Rezultat test skupa: 0.8957162240744331
    # boost(xt_train, xt_test, yt_train, yt_test, n_est=50, mode='save', rtype='cell')
    # Rezultat trening skupa: 1.0
    # Rezultat test skupa: 0.9696969696969697

    vot(x_train, x_test, y_train, y_test, power='hard', mode='save', rtype='gen')
    # Rezultat trening skupa: 0.9243166902052007
    # Rezultat test skupa: 0.9045357627447179
    # Najbolji odnos rezultata, veoma malo prilagodjavanje podacima
    vot(xt_train, xt_test, yt_train, yt_test, power='hard', mode='save', rtype='cell')
    # Rezultat trening skupa: 0.996996996996997
    # Rezultat test skupa: 0.9813519813519813

    """ Zakljucak: Iz ansambl metoda dobijamo veoma slicne rezultate, jedan od razloga je slican nacin rada,
        a takodje u ovim primerima je kao primaran model korisceno drvo odlucivanja, 
        jer se pokazalo do sada kao najbolje. Tako da za buduca ispitivanja cemo koristiti RandomForest metodu, 
        samo menjati njene parametre. Ova zapazanja ce se naravno naci u samom radu """


if __name__ == '__main__':
    main()
