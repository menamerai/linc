import re
from abc import ABC
from enum import Enum

import numpy as np
import torch
from lm import *
from logic import prove, get_all_variables
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    pipeline,
)

from pred_types import OWA_PRED


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
        max_new_tokens: int = 20,
    ) -> None:
        self.mode = mode
        self.device = device
        self.pg = pg
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, truncation_side="left"
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList(
                [StopOnWords(self.pg.stop_words, self.tokenizer, device)]
            ),
            pad_token_id=self.tokenizer.eos_token_id,
            # we can try beam search here, but i don't have the GPU for it
        )

    def predict(self, doc: dict[str, str | list[str]]) -> OWA_PRED:
        prompt = self.pg.generate(self.mode, doc)
        generation = self.generator(prompt)[0]["generated_text"]

        # get LAST element between <EVALUATE> tags using regex
        generation = re.findall(
            rf"<EVALUATE>\n*(.+?)\n*<\/EVALUATE>", generation, re.DOTALL
        )[-1]
        generation = generation.strip()
        if self.mode == MODEL_MODE.BASELINE:
            return self.evaluate_baseline(generation)
        elif self.mode == MODEL_MODE.NEUROSYMBOLIC:
            return self.evaluate_neurosymbolic(generation)

    def evaluate_baseline(result: str) -> OWA_PRED:
        if result.lower() == "true":
            return OWA_PRED.TRUE
        elif result.lower() == "false":
            return OWA_PRED.FALSE
        else:
            # OWA assume uncertain
            return OWA_PRED.UNK

    def evaluate_neurosymbolic(result: str) -> OWA_PRED:
        # TODO: not sure what to do here; are FOL expressions \n split?
        return OWA_PRED.UNK  # this is for you to implement


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
        max_new_tokens=50,
    )

    print(model.predict(example_doc))
