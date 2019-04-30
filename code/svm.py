import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import time

# read the JSON dataset
data = pd.read_json('posts.json')
data = data.sample(frac=1).reset_index(drop=True)

# removing posts which does not contain any body
data = data[data['body']!=""]

# see the distibution of the posts with respect to thier class
# data.groupby('class').count()['body'].plot(kind='bar',figsize= (8,8))

#split data
X = data['body']
Y = data['class']
# split the data into training, not training
train_X, rest_X, train_Label, rest_Label = train_test_split(X, Y, test_size=0.50)
# split the not training set  into testing and validation
test_X, validate_X , test_label, validate_label = train_test_split(rest_X, rest_Label, test_size=0.50)
# store the validation set in a seperate file for later use
validate_dataset = pd.concat([validate_X,validate_label],axis=1)
validate_dataset.to_csv('validation.csv')


# encode the labels
Encoder = LabelEncoder()
train_y = Encoder.fit_transform(train_Label)
test_y = Encoder.fit_transform(test_label)


# count vectorizer
count_vect = CountVectorizer(analyzer='word',lowercase=True, stop_words='english')
#count_vect = TfidfVectorizer(stop_words='english')
vectorizer = count_vect.fit(train_X)

# transform the data
train_X = vectorizer.transform(train_X)
test_X = vectorizer.transform(test_X)


# train the model
start = time.time()

clf = svm.SVC(kernel='linear')
clf.fit(train_X, train_y)

end = time.time()

print("Time taken train = ", end - start)

print()

'''# train the model
start = time.time()

clf = svm.SVC(kernel='rbf')
clf.fit(train_X, train_y)

end = time.time()

print("Time taken train = ", end - start)'''

predictions_SVM = clf.predict(train_X)
# calculate accuracy on training set
print("SVM Accuracy Score training set -> ", accuracy_score(predictions_SVM, train_y)*100)

predictions_SVM = clf.predict(test_X)
# calculate accuracy on testing set
print("SVM Accuracy Score testing set -> ", accuracy_score(predictions_SVM, test_y)*100)

validation = pd.read_csv('validation.csv')
validate_y = Encoder.fit_transform(validation['class'])
validate_x = vectorizer.transform(validation['body'])
predictions_SVM = clf.predict(validate_x)
# calculate accuracy on validation set
print("SVM Accuracy Score validation set -> ", accuracy_score(predictions_SVM, validate_y)*100)

rf = RandomForestRegressor(500)

start = time.time()

rf.fit(train_X, train_y)

end = time.time()

print("Time taken train = ", end - start)

start = time.time()

prediction_rf = rf.predict(train_X)

end = time.time()

print("Time taken train predict = ", end - start)

# Round off to 0 or 1 for accuracy
for index in range(len(prediction_rf)):
    prediction_rf[index] = int(round(prediction_rf[index]))

print("Random Forest Accuracy on training : ", accuracy_score(prediction_rf, train_y) * 100)

start = time.time()

prediction_rf = rf.predict(test_X)

end = time.time()

print("Time taken test predict = ", end - start)

# Round off to 0 or 1 for accuracy
for index in range(len(prediction_rf)):
    prediction_rf[index] = int(round(prediction_rf[index]))

print("Random Forest Accuracy on testing : ", accuracy_score(prediction_rf, test_y) * 100)

start = time.time()

prediction_rf = rf.predict(validate_x)

end = time.time()

print("Time taken validate predict = ", end - start)

# Round off to 0 or 1 for accuracy
for index in range(len(prediction_rf)):
    prediction_rf[index] = int(round(prediction_rf[index]))

print("Random Forest Accuracy on validation : ", accuracy_score(prediction_rf, validate_y) * 100)

