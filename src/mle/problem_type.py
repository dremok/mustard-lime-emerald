from enum import Enum


class ProblemType(Enum):
    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'
    CLUSTERING = 'clustering'
