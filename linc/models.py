import numpy as np

from lm import *
from logic import *
from abc import ABC
from enum import Enum


PRED = Enum("PRED", ["FALSE", "TRUE", "UNK", "ERR"])

class BaseModel(ABC):
    def __init__(self) -> None:
        raise NotImplementedError("initializer not implemented")
    
    def predict(s: str) -> PRED:
        raise NotImplementedError("prediction method not implemented")

class RandomModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        self.rng = np.random.default_rng(**kwargs)
        self.choices = [s for s in PRED]

    def predict(self, s:str)->PRED:
        return self.rng.choice(self.choices)
    
if __name__ == "__main__":
    model = RandomModel()
    [print(model.predict("s")) for _ in range(5)]