import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

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
clf = svm.SVC(gamma='auto')
clf.fit(train_X,train_y)

predictions_SVM = clf.predict(train_X)
# calculate accuracy on training set
print("SVM Accuracy Score training set -> ",accuracy_score(predictions_SVM, train_y)*100)

predictions_SVM = clf.predict(test_X)
# calculate accuracy on testing set
print("SVM Accuracy Score testing set -> ",accuracy_score(predictions_SVM, test_y)*100)



