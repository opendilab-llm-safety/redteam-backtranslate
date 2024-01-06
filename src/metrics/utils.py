from abc import ABC, abstractclassmethod
from typing import List, Text


class Metrics:

    @abstractclassmethod
    def compute(corpus: List[Text]):
        raise NotImplementedError
