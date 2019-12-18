from abc import ABC


class BaseTransformer(ABC):
    def transform(self, arr):
        raise NotImplementedError
