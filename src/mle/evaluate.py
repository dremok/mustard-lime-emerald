from abc import ABC

from sklearn.model_selection import cross_val_score


class Evaluator(ABC):
    def evaluate(self, X, y, **kwargs):
        raise NotImplementedError


class CrossValidation(Evaluator):
    def __init__(self, pipeline):
        self._pipeline = pipeline

    def evaluate(self, X, y, cv=5):
        scores = cross_val_score(self._pipeline, X, y, cv=cv)
        print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
