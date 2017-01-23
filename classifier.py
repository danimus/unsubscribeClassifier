#!/usr/bin/env python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('data/emails.csv', header=0, names=['unsubscribe', 'text'])

# Naive Bayes classifier for multinomial models
classifier = MultinomialNB(alpha=.05)

# Feature extraction
# min_df:
# When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
# This value is also called cut-off in the literature.
# If float, the parameter represents a proportion of documents, integer absolute counts.
# This parameter is ignored if vocabulary is not None.

# ngram_range:
# The lower and upper boundary of the range of n-values for different n-grams to be extracted.
# All values of n such that min_n <= n <= max_n will be used.
vectorizer = CountVectorizer(ngram_range=(1,2))

# Text from ['text'] column
traindata = df.text.astype(str)
# Apply vectorizer to training data
x_train = vectorizer.fit_transform(traindata)
# Labels from ['unsubscribe'] column
y_train = df.unsubscribe.replace([0, 1], ['is_not_unsubscribe', 'is_unsubscribe'])
# Train classifier
classifier.fit(x_train, y_train)

# Test data
is_unsubscribe = ["Please remove me your the mailing list"]
is_not_unsubscribe = ["Call me later"]

print "%s is classified as %s" % (is_unsubscribe, classifier.predict(vectorizer.transform(is_unsubscribe)))
print "%s is classified as %s" % (is_not_unsubscribe, classifier.predict(vectorizer.transform(is_not_unsubscribe)))