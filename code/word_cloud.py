import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from wordcloud import WordCloud, STOPWORDS

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

# Get all the bodies in one string

word_string_IAmA = ''

word_string_stories = ''

# Convert to string.
train_Label = list(train_Label)

train_X = list(train_X)

for body_index in range(len(train_X)):
    if str(train_Label[body_index]) == 'stories':
        word_string_stories += str(train_X[body_index]) + ' '
    else:
        word_string_IAmA += str(train_X[body_index]) + ' '

# Form word cloud

word_cloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=1200, height=1000).generate(word_string_IAmA)

plt.imshow(word_cloud)

plt.axis('off')

plt.show()

word_cloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=1200, height=1000).generate(word_string_stories)

plt.imshow(word_cloud)

plt.axis('off')

plt.show()
