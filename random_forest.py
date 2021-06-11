#!/usr/bin/python

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ml_helpers import evaluate_model

# Will test n_estimators and min_samples_split
def best_random_forest(n_estimators_list, min_samples_splits, random_state, features_train, features_test, labels_train, labels_test):
    print()
    print("RANDOM FOREST CLASSIFIER")
    print()
    top_accuracy = 0
    clf = RandomForestClassifier(random_state=random_state)
    clf_final = RandomForestClassifier(random_state=random_state)
    final_params = {"n_estimators": 100, "min_samples_split": 2}
    for n_estimators in n_estimators_list:
        print(f"n_estimators = {n_estimators}---------------------------------")
        for min_samples_split in min_samples_splits:
            print(f"min_samples_split = {min_samples_split}---------------------------------")
            clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
            training_time, predict_time, accuracy = evaluate_model(clf, features_train, features_test, labels_train, labels_test)
            print(f"n_estimators = {n_estimators}")
            print(f"min_samples_split = {min_samples_split}")
            print(f"Training time: {round(training_time, 3)}s")
            print(f"Prediction time: {round(predict_time, 3)}s")
            print(f"Accuracy: {accuracy}")
            print()
            if accuracy > top_accuracy:
                top_accuracy = accuracy
                clf_final = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
                final_params["n_estimators"] = n_estimators
                final_params["min_samples_split"] = min_samples_split

    final_params["accuracy"] = top_accuracy
    return clf_final, final_params