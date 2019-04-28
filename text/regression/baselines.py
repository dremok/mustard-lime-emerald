from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def tfidf_logreg(train, test, text_col, target_col):
    train['class'] = train[target_col] >= 0.5

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression()),
    ])

    scores = cross_val_score(text_clf, train[text_col], train['class'], cv=3)
    mean_clf_score = scores.mean()

    text_clf.fit(train[text_col], train['class'])

    predicted = text_clf.predict_proba(test[text_col])

    return predicted, mean_clf_score
