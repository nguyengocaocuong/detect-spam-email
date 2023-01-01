import pandas as pd
import os
import joblib
from verify_data  import *

def main():

    if not os.path.exists("./data/model_detect_spam_email_NB.joblib"):
        print("You should be run file train_model_detect_email.py first.")
        return

    email_number = -1
    model = joblib.load("./data/model_detect_spam_email_NB.joblib")
    while True:
        list_files = os.listdir("./data/test_email/")
        for i in range(len(list_files)):
            print(f"{i+1}. {list_files[i]}")
        print("0. Exist")
        print("Enter you email number: ")
        email_number = int(input())
        if email_number == 0:
            break
        email = ""
        with  open(f"./data/test_email/{list_files[email_number-1]}") as file:
            email = file.read()
        matrixs = prepare_email(email)
        y_pred = model.predict(matrixs)
        print("===================")
        print(f"Input : {list_files[email_number-1]}")
        if y_pred[0] == 0:
            print("\tResult: Ham email")
        else:
            print("\tResult: Spam email")
        print("===================")


main()