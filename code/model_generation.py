__author__ = 'Ketan Kokane'
__author__ = 'Ameya Nagnur'
"""
below code generates the 2 different models SVM and Random Forest Classifier 
to do so, we split the data into training dev and test. 
We also preprocess the dataset to remove punctuations, stop words from the text. 
Then we use Count Vectorizer to convert the text data into feature vectors.  
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time


def read_data_set():
    """
    Helper function to read the whole available dataset
    :return:
    """
    global data
    # read the JSON dataset
    data = pd.read_json('posts.json')
    data = data.sample(frac=1).reset_index(drop=True)
    # removing posts which does not contain any body
    data = data[data['body'] != ""]


def split_data_set():
    """
    Helper function split the data into 3 parts of train, dev and test with 50- 25-25 ratio

    :return:
    """
    global train_X, train_Label, test_X, test_label
    # split data
    X = data['body']
    Y = data['class']
    # split the data into training, not training
    train_X, rest_X, train_Label, rest_Label = train_test_split(X, Y, test_size=0.50)
    # split the not training set  into testing and validation
    test_X, validate_X, test_label, validate_label = train_test_split(rest_X, rest_Label,
                                                                      test_size=0.50)
    # see the distibution of the posts with respect to thier class
    # data.groupby('class').count()['body'].plot(kind='bar',figsize= (8,8))

    # store the validation set in a seperate file for later use
    validate_dataset = pd.concat([validate_X, validate_label], axis=1)
    validate_dataset.to_csv('validation.csv')


def encode_labels(Y_Label):
    """
    Uses a already Fitted Encoder to encode the labels
    i.e to convert class like 'iAmA' and 'stories' to 0 and 1
    :param Y_Label:
    :return:
    """
    return Encoder.transform(Y_Label)


def convert_to_vector(X):
    """
    Convert the given attribute to a vector by using CountVectorizer
    :param X:
    :return:
    """
    return vectorizer.transform(X)


def train_model(classfier, X, Y, text):
    """
    Generic helper function which trains the model for given classifer and X attributes
    Y label
    :param classfier:
    :param X:
    :param Y:
    :return:
    """
    # train the model
    start = time.time()
    classfier.fit(X, Y)
    end = time.time()
    print("Time taken to train ",text, "= ", end - start)
    return classfier


def print_accuracy(model, x, y, text):
    """
    Helper function which given a model, prints  its accuracy for the given data set
    :param model:
    :param x:
    :param y:
    :param text:
    :return:
    """
    pred_y = model.predict(x)
    # calculate accuracy on training set
    print(text, accuracy_score(pred_y, y) * 100)


def get_validation_set():
    """
    Helper function which reads the validation dataset from a file
    does the required preprocessing on it and return the vectors and labels
    :return:
    """
    validation = pd.read_csv('validation.csv')
    validate_y = encode_labels(validation['class'])
    validate_x = convert_to_vector(validation['body'])
    return validate_y, validate_x


def accuracy_on_all_set_of_data(model,text):
    """
    Helper function to print the required accuracies on all the data set
    :param model:
    :return:
    """
    print_accuracy(model, train_X, train_y, text+" Accuracy Score training set -> ")

    test_y = encode_labels(test_label)

    print_accuracy(model, test_X, test_y, text+" Accuracy Score testing set ->")

    y, x = get_validation_set()

    print_accuracy(model, x, y, text+" Accuracy Score validation set -> ")


def main():
    global Encoder, train_y, vectorizer, train_X, test_X
    read_data_set()
    split_data_set()
    # encode the labels
    Encoder = LabelEncoder()
    Encoder.fit(train_Label)
    train_y = encode_labels(train_Label)
    # count vectorizer
    count_vect = CountVectorizer(analyzer='word', lowercase=True, stop_words='english')
    # count_vect = TfidfVectorizer(stop_words='english')
    vectorizer = count_vect.fit(train_X)
    # transform the data
    train_X = convert_to_vector(train_X)
    test_X = convert_to_vector(test_X)
    clf = train_model(svm.SVC(kernel='linear'), train_X, train_y, "SVM")
    accuracy_on_all_set_of_data(clf, "SVM")
    rf = train_model(RandomForestClassifier(100), train_X, train_y, "RandomForest")
    accuracy_on_all_set_of_data(rf, "Random Forest")


main()
