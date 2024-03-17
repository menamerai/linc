import re
from abc import ABC
from enum import Enum

import numpy as np
import torch
from lm import *
from logic import *
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

OWA_PRED = Enum("PRED", ["FALSE", "TRUE", "UNK"])


class BaseModel(ABC):
    def __init__(self) -> None:
        raise NotImplementedError("initializer not implemented")

    def predict(s: str) -> OWA_PRED:
        raise NotImplementedError("prediction method not implemented")


class RandomModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        self.rng = np.random.default_rng(**kwargs)
        self.choices = [s for s in OWA_PRED]

    def predict(self, _: str) -> OWA_PRED:
        return self.rng.choice(self.choices)


class HFModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        mode: MODEL_MODE,
        pg: PromptGenerator,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, truncation_side="left"
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=4096,
            device=device,
        )
        self.mode = mode
        self.pg = pg
        self.device = device

    def predict(self, doc: dict[str, str | list[str]]) -> OWA_PRED:
        if self.mode == MODEL_MODE.BASELINE:
            return self.predict_baseline(doc)
        elif self.mode == MODEL_MODE.NEUROSYMBOLIC:
            return self.predict_neurosymbolic(doc)

    def predict_baseline(self, doc: dict[str, str | list[str]]) -> OWA_PRED:
        prompt = self.pg.generate(self.mode, doc)
        generation = self.generator(prompt)[0]["generated_text"]
        # get everything between <EVALUATE> tags using regex
        generation = re.search(
            rf"{self.pg.container[0]}(.*?){self.pg.container[1]}", generation, re.DOTALL
        ).group(1)
        generation = generation.strip()

        if generation.lower() == "true":
            return OWA_PRED.TRUE
        elif generation.lower() == "false":
            return OWA_PRED.FALSE
        else:
            # OWA assume uncertain
            return OWA_PRED.UNK


if __name__ == "__main__":
    example_doc = {
        "premises": [
            "If a city hold a Summer Olympics, and the city is a US city, then the Summer Olympics will be in the US.",
            "If a city is in a state which is in US, the city is a US city.",
            "If a city in a state, and a Summer Olympics is in this city, then the Summer Olympics is in this state.",
            "The 2028 Summer Olympics is scheduled to take place in Los Angeles(LA).",
            "LA is a city in California(CA).",
            "Atlanta is a US city.",
            "Atlanta is in Georgia(CA).",
            "CA is a state in the United States.",
            "Boxing, modern pentathlon, and weightlifting will be removed from The 2028 Summer Olympics.",
            "Atlanta in the United States hold the 1996 Summer Olympics.",
        ],
        "conclusion": "The 2028 Summer Olympics will take place in the US.",
        "premises_FOL": [
            "∀x ∀y (LaLiga(x) ∧ LaLiga(y) ∧ MorePoints(x, y) → HigherRank(x, y))",
            "∀x ∀y (LaLiga(x) ∧ LaLiga(y) ∧ ¬MorePoints(x, y) ∧ ¬MorePoints(y, x) ∧ MorePointsInGameBetween(x, y) → HigherRank(x, y))",
            "LaLiga(realMadrid) ∧ LaLiga(barcelona)",
            "MorePoints(realMadrid, barcelona)",
            "¬MorePointsInGameBetween(realMadrid, barcelona) ∧ ¬MorePointsInGameBetween(barcelona, realMadrid)",
        ],
    }

    model = HFModel(
        "microsoft/phi-2",
        "microsoft/phi-2",
        MODEL_MODE.BASELINE,
        PromptGenerator(),
    )

    print(model.predict(example_doc))
