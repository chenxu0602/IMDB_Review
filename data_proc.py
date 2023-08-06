import os, sys
import numpy as np
import pandas as pd
import re, json, nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

df = pd.read_csv("movie_data.csv", encoding="utf-8")

count = CountVectorizer()
tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
np.set_printoptions(precision=2)

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

df["review"] = df["review"].apply(preprocessor)


def tokenizer(text): return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


nltk.download("stopwords")
stop = stopwords.words("english")

X_train = df.loc[:25000, "review"].values
y_train = df.loc[:25000, "sentiment"].values
X_test = df.loc[25000:, "review"].values
y_test = df.loc[25000:, "sentiment"].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

small_param_grid = [
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [None],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "clf__penalty": ["l2"],
        "clf__C": [1.0, 10.0]
    },
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [stop, None],
        "vect__tokenizer": [tokenizer],
        "vect__use_idf": [False],
        "vect__norm": [None],
        "clf__penalty": ["l2"],
        "clf__C": [1.0, 10.0]
    },
]

lr_tfidf = Pipeline([
    ("vect", tfidf),
    ("clf", LogisticRegression(solver="liblinear"))
])

gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, scoring="accuracy", cv=5, verbose=2, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

print(f"CV Accuracy: {gs_lr_tfidf.best_score_:.3f}")

clf = gs_lr_tfidf.best_estimator_
print(f"Test Accuracy: {clf.score(X_test, y_test):.3f}")
