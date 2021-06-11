#!/usr/bin/python

import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from ml_helpers import evaluate_model

# Will explore n_estimators, learning_rate and must maintain random_state for comparable results
def best_adaboost(n_estimators_list, learning_rates, random_state, features_train, features_test, labels_train, labels_test):
    print()
    print("ADABOOST CLASSIFIER")
    print()
    top_accuracy = 0
    clf = AdaBoostClassifier(random_state=random_state)
    clf_final = AdaBoostClassifier(random_state=random_state)
    final_params = {"n_estimators": 50, "learning_rate": 1}
    for n_estimators in n_estimators_list:
        print(f"n_estimators = {n_estimators}---------------------------------")
        for learning_rate in learning_rates:
            print(f"learning_rate = {learning_rate}---------------------------------")
            clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
            training_time, predict_time, accuracy = evaluate_model(clf, features_train, features_test, labels_train, labels_test)
            print(f"n_estimators = {n_estimators}")
            print(f"learning_rate = {learning_rate}")
            print(f"Training time: {round(training_time, 3)}s")
            print(f"Prediction time: {round(predict_time, 3)}s")
            print(f"Accuracy: {accuracy}")
            print()
            if accuracy > top_accuracy:
                top_accuracy = accuracy
                clf_final = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
                final_params["n_estimators"] = n_estimators
                final_params["learning_rate"] = learning_rate

    final_params["accuracy"] = top_accuracy
    return clf_final, final_params
