
'''
    "Character Identification using NB, DT & SVM from Scikit-learn"

    The following 3 Machine Learning algorithms run on 2 datasets, located in "DataSet_Release 2" folder.
        1. Decision Tree (DT)
        2. Naive Bayes classifier (NB) - BernoulliNB
        3. Support Vector Machine (SVM)

    Datasets located in "DataSet_Release 2" folder.
    Dataset 1 contains images of the 26 uppercase letters [A – Z] and 25 lowercase letters [a – z]
    Dataset 2 contains images of 10 Greek letters.

    The program generates one output file for each of the 3 models and each of the validation and test sets.
'''


import csv
import pickle
import os

from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB, GaussianNB
from sklearn import svm, metrics, tree

relativePath = os.path.dirname(os.path.abspath(__file__))


def testAndLoadModelsForDS1(X_test, X_train, y_test, y_train):
    testLoadingModel("ds1Test-nb.csv", X_train, y_train, X_test, y_test)
    testLoadingModel("ds1Test-dt.csv", X_train, y_train, X_test, y_test)
    testLoadingModel("ds1Test-3.csv", X_train, y_train, X_test, y_test)


def testAndLoadModelsForDS2(X_test, X_train, y_test, y_train):
    testLoadingModel("ds1Test-nb.csv", X_train, y_train, X_test, y_test)
    testLoadingModel("ds1Test-dt.csv", X_train, y_train, X_test, y_test)
    testLoadingModel("ds1Test-3.csv", X_train, y_train, X_test, y_test)


def main():
    map = readCharacterMappingInfo(os.path.join(relativePath, "DataSet-Release 2", "ds1", "ds1Info.csv"))
    X_train, y_train = readCSVForXAndY(map, os.path.join(relativePath, 'DataSet-Release 2', 'ds1', 'ds1Train.csv'))
    x_validation, y_validation = readCSVForXAndY(map,
                                                 os.path.join(relativePath, 'DataSet-Release 2', 'ds1', 'ds1Val.csv'))
    X_test, y_test = readCSVForXAndYForTest(map, os.path.join(relativePath, 'DataSet-Release 2', 'ds1', 'ds1Test.csv'))
    # calculate models for DS1
    calculateModels(X_test, X_train, map, x_validation, y_test, y_train, y_validation)
    # testAndLoadModelsForDS1(X_test, X_train, y_test, y_train)

    map = readCharacterMappingInfo(os.path.join(relativePath, "DataSet-Release 2", "ds2", "ds2Info.csv"))
    X_train, y_train = readCSVForXAndY(map, os.path.join(relativePath, 'DataSet-Release 2', 'ds2', 'ds2Train.csv'))
    x_validation, y_validation = readCSVForXAndY(map,
                                                 os.path.join(relativePath, 'DataSet-Release 2', 'ds2', 'ds2Val.csv'))
    X_test, y_test = readCSVForXAndYForTest(map, os.path.join(relativePath, 'DataSet-Release 2', 'ds2', 'ds2Test.csv'))
    # calculate models for DS2
    calculateModelsForDS2(X_test, X_train, map, x_validation, y_test, y_train, y_validation)
    #testAndLoadModelsForDS2(X_test, X_train, y_test, y_train)


def saveModel(model, filename):
    with open('Output/' + filename + '.pkl', 'wb') as file:
        pickle.dump(model, file)


def openModel(path):
    fullPath = os.path.join(relativePath, "output", path + ".pkl")
    file = open(fullPath, 'rb')
    response = pickle.load(file)
    file.close()
    return response


def testLoadingModel(path, X_train, y_train, X_test, y_test):
    model = openModel(path)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nModel loaded the file:" + path + ", is loaded")


def calculateModels(X_test, X_train, map, x_validation, y_test, y_train, y_validation):
    trainNaiveBayes(x_validation, X_train, y_validation, y_train, map, "ds1Val-nb.csv", False)
    trainNaiveBayes(X_test, X_train, y_test, y_train, map, "ds1Test-nb.csv", True)
    trainDecisionTreeClassifier(x_validation, X_train, y_validation, y_train, map, "ds1Val-dt.csv", False)
    trainDecisionTreeClassifier(X_test, X_train, y_test, y_train, map, "ds1Test-dt.csv", True)
    trainSVM(x_validation, X_train, y_validation, y_train, map, "ds1Val-3.csv", False)
    trainSVM(X_test, X_train, y_test, y_train, map, "ds1Test-3.csv", True)


def calculateModelsForDS2(X_test, X_train, map, x_validation, y_test, y_train, y_validation):
    trainNaiveBayes(x_validation, X_train, y_validation, y_train, map, "ds2Val-nb.csv", False)
    trainNaiveBayes(X_test, X_train, y_test, y_train, map, "ds2Test-nb.csv", True)
    trainDecisionTreeClassifier(x_validation, X_train, y_validation, y_train, map, "ds2Val-dt.csv", False)
    trainDecisionTreeClassifier(X_test, X_train, y_test, y_train, map, "ds2Test-dt.csv", True)
    trainSVM(x_validation, X_train, y_validation, y_train, map, "ds2Val-3.csv", False)
    trainSVM(X_test, X_train, y_test, y_train, map, "ds2Test-3.csv", True)



def readCharacterMappingInfo(filepath):
    map = {}
    # with open('DataSet-Release 1\ds1\ds1Info.csv') as csvDataFile:
    with open(filepath) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            map[row[0]] = row[1]
    return map


def readCSVForXAndY(map, fileName):
    X_train = []
    y_train = []
    with open(fileName) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            y_train.append(map[row[len(row) - 1]])
            row = row[:-1]
            X_train.append(np.array(row).astype(np.float))
    return X_train, y_train


def readCSVForXAndYForTest(map, fileName):
    X_train = []
    y_train = []
    with open(fileName) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            y_train.append(map[row[len(row) - 1]])
            X_train.append(np.array(row).astype(np.float))
    return X_train, y_train


def trainNaiveBayes(X_test, X_train, y_test, y_train, map, fileName, isTest):
    from sklearn.metrics import accuracy_score
    model = BernoulliNB()
    # model = MultinomialNB()
    # model = GaussianNB()
    # model = ComplementNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if (isTest):
        saveModel(model, fileName)
    else:
        print("\nAccuracy of the test data using Naive Bayes model is:" + str(accuracy_score(y_pred, y_test)) + "%")
    # print("\nClassification report for %s:\n%s\n"
    #       % (model, metrics.classification_report(y_test, y_pred)))
    # print("Confusion matrix for Naive Bayes model:\n%s" % metrics.confusion_matrix(y_test, y_pred))
    with open(fileName, 'w') as file:
        csvWrite = csv.writer(file, delimiter=',')
        for i in range(len(y_pred)):
            val = getKeyFromMap(map, y_pred[i])
            row = [int(i + 1), +int(val)]
            csvWrite.writerow(row)


def trainDecisionTreeClassifier(X_test, X_train, y_test, y_train, map, fileName, isTest):
    from sklearn.metrics import accuracy_score
    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if (isTest):
        saveModel(model, fileName)
    else:
        print("\nAccuracy of the test data using Decision Tree model is:" + str(accuracy_score(y_pred, y_test)) + "%")
    # print("\nClassification report for %s:\n%s\n"
    #       % (model, metrics.classification_report(y_test, y_pred)))
    # print("Confusion matrix for Decision Tree model:\n%s" % metrics.confusion_matrix(y_test, y_pred))
    with open(fileName, 'w') as file:
        csvWrite = csv.writer(file, delimiter=',')
        for i in range(len(y_pred)):
            val = getKeyFromMap(map, y_pred[i])
            row = [int(i + 1), +int(val)]
            csvWrite.writerow(row)


def trainSVM(X_test, X_train, y_test, y_train, map, fileName, isTest):
    clf = svm.SVC(kernel='rbf', gamma=0.01, C=1000)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if (isTest):
        saveModel(model, fileName)
    else:
        print("\nAccuracy of the test data using SVM model is:" + str(accuracy_score(y_pred, y_test)) + "%")
    # print("\nClassification report for %s:\n%s\n"
    #       % (model, metrics.classification_report(y_test, y_pred)))
    # print("Confusion matrix for SVM model:\n%s" % metrics.confusion_matrix(y_test, y_pred))
    with open(fileName, 'w') as file:
        csvWrite = csv.writer(file, delimiter=',')
        for i in range(len(y_pred)):
            val = getKeyFromMap(map, y_pred[i])
            row = [int(i + 1), +int(val)]
            csvWrite.writerow(row)


def getKeyFromMap(dictionary, val):
    for key, value in dictionary.items():
        if value == val:
            return (key)


if __name__ == "__main__":
    main()
