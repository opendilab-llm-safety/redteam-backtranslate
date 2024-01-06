from abc import ABC, abstractclassmethod
from typing import List, Text


class Metrics(ABC):

    @abstractclassmethod
    def compute(corpus: List[Text]):
        raise NotImplementedError
