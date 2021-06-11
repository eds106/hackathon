#!/usr/bin/python

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ml_helpers import evaluate_model

def best_svc(kernels, Cs, gammas, features_train, features_test, labels_train, labels_test):
    print()
    print("SUPPORT VECTOR CLASSIFIER")
    print()
    top_accuracy = 0
    svc = SVC()
    svc_final = SVC()
    final_params = {"kernel": "rbf", "C": 1, "gamma": "scale"}
    for kernel in kernels:
        print(f"Kernel = {kernel}---------------------------------")
        for c in Cs:
            print(f"C = {c}---------------------------------")
            for index, gamma in enumerate(gammas):
                # Gamma doesn't affect linear kernels; only do one iteration for linear kernels
                if kernel == "linear" and index != 0:
                    continue

                svc = SVC(kernel=kernel, C=c, gamma=gamma)
                training_time, predict_time, accuracy = evaluate_model(svc, features_train, features_test, labels_train, labels_test)
                print(f"Kernel = {kernel}")
                print(f"C = {c}")
                print(f"gamma = {gamma}")
                print(f"Training time: {round(training_time, 3)}s")
                print(f"Prediction time: {round(predict_time, 3)}s")
                print(f"Accuracy: {accuracy}")
                print()
                if accuracy > top_accuracy:
                    top_accuracy = accuracy
                    svc_final = SVC(kernel=kernel, C=c, gamma=gamma)
                    final_params["kernel"] = kernel
                    final_params["C"] = c
                    final_params["gamma"] = gamma

    final_params["accuracy"] = top_accuracy
    return svc_final, final_params
