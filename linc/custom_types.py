from enum import Enum
from functools import total_ordering

@total_ordering
class OWA_PRED(Enum):
    FALSE = 0
    TRUE = 1
    UNK = 2
    ERR = 3
    
    def __lt__(self, a):
        if type(self) == type(a):
            return self.value < a.value
        raise ValueError(f"cannot compare enum to type {type(a)}")

MODEL_MODE = Enum("MODEL_MODE", ["BASELINE", "NEUROSYMBOLIC"])
