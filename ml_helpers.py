from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from time import time
import numpy as np

def train_model(model_class, X_train, y_train):
    trained_model = model_class()
    trained_model.fit(X_train, y_train)
    return trained_model

def evaluate_model(model, features_train, features_test, labels_train, labels_test):
    t0 = time()
    model.fit(features_train, labels_train)
    training_time = time() - t0
    t1 = time()
    labels_predict = model.predict(features_test)
    predict_time = time() - t1
    accuracy = accuracy_score(labels_test, labels_predict)
    return training_time, predict_time, accuracy

def regression_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r_squared = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    return r_squared, mse, rmse, mae


def classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    #TODO: ROC and AUC should be added
    return accuracy, precision, recall


def classification_analysis(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, report, confusion


def generate_linear():
    raise NotImplementedError


def generate_quadratic():
    raise NotImplementedError


def generate_polynomial():
    raise NotImplementedError


def generate_classes():
    raise NotImplementedError
