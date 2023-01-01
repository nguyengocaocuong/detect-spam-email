from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from statistics import mean
import numpy as np
import pandas as pd
import os
import joblib
from verify_data import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def train_model_and_save(input_emails):
    matrixs = []
    labels = []
    if not os.path.exists("./data/matrixs.npy") or not os.path.exists("./data/labels.npy"):
        input_emails = pd.read_csv("./data/spam_ham_dataset.csv").to_numpy()
        matrixs, labels = create_matrixs(input_emails)

        np.save('./data/matrixs.npy', matrixs)
        np.save('./data/labels.npy', labels)
    
    matrixs = np.load("./data/matrixs.npy")
    labels = np.load("./data/labels.npy")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    lst_accu_stratified = []
    multinomialNB = MultinomialNB()
    max = []
    x = 0
    print("Start training model detect email....")
    while True:
        try:
            for train_index, test_index in skf.split(matrixs, labels):
                x_train, x_test = matrixs[train_index], matrixs[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                multinomialNB.fit(x_train, y_train)
                sccore = multinomialNB.score(x_test, y_test)
                if x == 0 or x < sccore:
                    max = []
                    x = sccore
                    max.append(x_train)
                    max.append(y_train)
                    max.append(x_test)
                    max.append(y_test)
                lst_accu_stratified.append(sccore)
            break
        except:
            print("Step ")
            continue
    
    print(lst_accu_stratified)
    print("Mean score: ",mean(lst_accu_stratified))
    print("Done train....")
    
    print("Start test model...")
    multinomialNB = multinomialNB.fit(max[0], max[1])
    y_pred = multinomialNB.predict(max[2])
    cr_m = confusion_matrix(max[3], y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cr_m, display_labels = ["Ham", "Spam"])
    cm_display.plot()
    plt.show()
    print("Done test ...")
    joblib.dump(multinomialNB, f"./data/model_detect_spam_email_NB.joblib")
input_emails = pd.read_csv("./data/spam_ham_dataset.csv")
train_model_and_save(input_emails.to_numpy())
