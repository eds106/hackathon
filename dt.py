#!/usr/bin/python

from sklearn.tree import DecisionTreeClassifier
from ml_helpers import evaluate_model

def best_dt(min_splits, random_state, features_train, features_test, labels_train, labels_test):
    print()
    print("DECISION TREE CLASSIFIER")
    print()
    dt = DecisionTreeClassifier(random_state=random_state)
    dt_final = DecisionTreeClassifier()
    top_accuracy = 0
    final_params = {}
    for min_split in min_splits:
        dt.min_samples_split = min_split
        training_time, predict_time, accuracy = evaluate_model(dt, features_train, features_test, labels_train, labels_test)
        print(f"min_samples_split = {min_split}-----------------")
        print(f"Training time: {round(training_time, 3)}s")
        print(f"Prediction time: {round(predict_time, 3)}s")
        print(f"Accuracy: {accuracy}")
        print()
        if accuracy > top_accuracy:
            top_accuracy = accuracy
            dt_final.min_samples_split = min_split
            final_params["min_samples_split"] = min_split
    final_params["accuracy"] = top_accuracy
    return dt_final, final_params
