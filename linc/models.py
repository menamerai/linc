import re
from abc import ABC
from enum import Enum

import cohere
import google.generativeai as genai
import numpy as np
import torch
from lm import *
from logic import get_all_variables, prove
from pred_types import OWA_PRED
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    pipeline,
)


class BaseModel(ABC):
    def __init__(self) -> None:
        raise NotImplementedError("initializer not implemented")

    def predict(s: str) -> OWA_PRED:
        raise NotImplementedError("prediction method not implemented")

    def evaluate_baseline(self, result: str) -> OWA_PRED:
        if result.lower() == "true":
            return OWA_PRED.TRUE
        elif result.lower() == "false":
            return OWA_PRED.FALSE
        else:
            # OWA assume uncertain
            return OWA_PRED.UNK

    def evaluate_neurosymbolic(self, result: str) -> OWA_PRED:
        return OWA_PRED.UNK  # this is for you to implement


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
        max_new_tokens: int = 50,
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
        # TODO: write different regex for neurosymbolic mode
        generation = re.findall(
            rf"<EVALUATE>\n*(.+?)\n*<\/EVALUATE>", generation, re.DOTALL
        )[-1]
        generation = generation.strip()
        if self.mode == MODEL_MODE.BASELINE:
            return self.evaluate_baseline(generation)
        elif self.mode == MODEL_MODE.NEUROSYMBOLIC:
            return self.evaluate_neurosymbolic(generation)


class GeminiModel(BaseModel):
    def __init__(
        self,
        google_api_key: str,
        pg: PromptGenerator,
        mode: MODEL_MODE,
        model_name: str = "gemini-pro",
        max_new_tokens: int = 50,
    ) -> None:
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.pg = pg
        self.mode = mode
        self.max_new_tokens = max_new_tokens

    def predict(self, doc: dict[str, str | list[str]]) -> OWA_PRED:
        prompt = self.pg.generate(self.mode, doc)
        generation = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,  # we can increase this later
                max_output_tokens=self.max_new_tokens,
                # the stop sequences are NOT included in the output
                stop_sequences=["</EVALUATE>"],
            ),
        )
        text = generation.text  # might be different for multiple candidates
        text = text.strip()
        # since the generation stops at </EVALUATE>, and does not include the prompt
        # just generation.text is exactly what we need
        if self.mode == MODEL_MODE.BASELINE:
            return self.evaluate_baseline(text)
        elif self.mode == MODEL_MODE.NEUROSYMBOLIC:
            return self.evaluate_neurosymbolic(text)


class CohereModel(BaseModel):
    def __init__(
        self,
        api_key: str,
        pg: PromptGenerator,
        mode: MODEL_MODE,
        model_name: str = "command",
        max_new_tokens: int = 50,
    ) -> None:
        self.pg = pg
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.model = cohere.Client(api_key=api_key)
        self.model_name = model_name

    def predict(self, doc: dict[str, str | list[str]]) -> OWA_PRED:
        prompt = self.pg.generate(self.mode, doc)
        # generate is legacy, cohere is asking us to use chat instead
        # but chat doesn't have a stop_sequences parameter
        # read more here https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat
        generation = self.model.generate(
            prompt,
            end_sequences=["</EVALUATE>"],
            max_tokens=self.max_new_tokens,
            num_generations=1,
            model=self.model_name,
        )

        text = generation[0].text
        text = text.strip()
        if self.mode == MODEL_MODE.BASELINE:
            return self.evaluate_baseline(text)
        elif self.mode == MODEL_MODE.NEUROSYMBOLIC:
            return self.evaluate_neurosymbolic(text)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

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
            "∀x ∀y (SummerOlympicsCity(x, y) ∧ CityInCountry(y, us) → SummerOlympicsCountry(x, us))",
            "∀x ∀y (CityInState(x, y) ∧ StateInCountry(y, us) → CityInCountry(x, us))",
            "∀x ∀y ∀z (CityInState(x, y) ∧ SummerOlympicsCity(z, x) → SummerOlympicsState(z, y))",
            "SummerOlympicsCity(y2028, la)",
            "CityInState(la, ca)",
            "CityInCountry(atlanta, us)",
            "StateInCountry(ca, us)",
            "CityInState(atlanta, ga)",
            "¬InSummerOlympics(y2028, boxing) ∧ ¬InSummerOlympics(y2028, modern_pentathlon) ∧ ¬InSummerOlympics(y2028, weightlifting)",
            "SummerOlympicsCity(y1996, atlanta)",
        ],
        # "label": "True",
    }

    """ model = HFModel(
        "microsoft/phi-2",
        "microsoft/phi-2",
        MODEL_MODE.BASELINE,
        PromptGenerator(),
        max_new_tokens=50,
    )

    print(model.predict(example_doc)) """

    """ model = GeminiModel(
        os.getenv("GOOGLE_API_KEY"), PromptGenerator(), MODEL_MODE.BASELINE
    )

    print(model.predict(example_doc)) """

    model = CohereModel(
        os.getenv("COHERE_API_KEY"), PromptGenerator(), MODEL_MODE.BASELINE
    )

    print(model.predict(example_doc))
