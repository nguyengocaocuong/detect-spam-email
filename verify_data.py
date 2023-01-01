from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import os
import nltk
nltk.download('punkt')

def extract_feature(input_emails):
    stopwords_ = stopwords.words("english")
    porterStemmer = PorterStemmer()
    count_words = {}
    for i in range(len(input_emails)):
        email = input_emails[i]
        words = [w for w in word_tokenize(email[2][9:]) if not w in stopwords_]
        words = [porterStemmer.stem(w) for w in words]

        for word in words:
            if word.isalpha() and len(word) > 1:
                if count_words.get(word):
                    count_words[word] += 1
                else:
                    count_words[word] = 1
   
    set_words =  [key for key in count_words.keys() if count_words[key] > 1]
    with open("./data/set_words.txt", "w") as file:
        json.dump(set_words,file)
    print("Number of Feature: ",len(set_words))

def create_matrixs(input_emails):
    matrixs = []
    labels = []
    set_words = []
    stopwords_ = stopwords.words("english")
    porterStemmer = PorterStemmer()
    if not os.path.exists("./data/set_words.txt"):
        extract_feature(input_emails)
    with open("./data/set_words.txt") as file:
        set_words = json.load(file)
    for i in range(len(input_emails)):
        email = input_emails[i]
        words = [w for w in word_tokenize(email[2][9:]) if not w in stopwords_]
        vector = [0 for i in range(len(set_words))]

        for w in words:
            w = porterStemmer.stem(w)
            if w in set_words:
                vector[set_words.index(w)] += 1
        matrixs.append(vector)
        labels.append(email[-1])
    return matrixs, labels

def prepare_email(email):
    matrixs = []
    set_words = []
    stopwords_ = stopwords.words("english")
    porterStemmer = PorterStemmer()
    with open("./data/set_words.txt") as file:
        set_words = json.load(file) 
    words = [w for w in word_tokenize(email[9:]) if not w in stopwords_]
    vector = [0 for i in range(len(set_words))]
    for w in words:
        w = porterStemmer.stem(w)
        if w in set_words:
            vector[set_words.index(w)] += 1
    matrixs.append(vector)
    return matrixs