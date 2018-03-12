import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# by giving the parameter quoting=3, we avoid double quotes in the reviews
dataset = pd.read_csv(
    "Restaurant_Reviews.tsv", delimiter='\t',
    quoting=3)

#  cleaning the text
import re
import nltk

# downloading list of words that we want to remove from the review such as this, at
nltk.download('stopwords')
# import stopwords
from nltk.corpus import stopwords

# getting root of the word, so that we can reduce the index of the sparse matrix
# avoiding have irrelevant huge data
from nltk.stem.porter import PorterStemmer

# creating an instance to Stemmer class
ps = PorterStemmer()
# corpus is a common word in natural language processing
corpus = []
for i in range(0, 1000):
    # only keeping the text, exclude numbers, marks etc
    # only A-Z
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # putting all the characters to lower case
    review = review.lower()

    # remove the words that do not help the algorithm
    # for example : this, at
    # the review was a string, to eliminate those words we have to convert to a list
    review = review.split()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    # converting the review which is in the type of list to the string type back
    review = ' '.join(review)

    # end of the cl

    corpus.append(review)

# Creating the Bag of Words model
# columns for each words
# rows are reviews
# the number of the repetition of the word

# tokenization, is taking all the different words of the review,
# creating one column for each of these words
# the columns are also independent variables
from sklearn.feature_extraction.text import CountVectorizer

countVectorizer = CountVectorizer(max_features=1500)
X = countVectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
# end of the bag of words

# Train the model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)

MyPrediction = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, MyPrediction)

# %73 accuracy