import re
from abc import ABC

import cohere
import google.generativeai as genai
import numpy as np
from custom_types import *
from lm import *
from logic import get_all_variables, prove
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
        config: HFModelConfig,
    ) -> None:
        self.config = config
        self.pg = config.pg
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=config.q_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, truncation_side="left"
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
            stopping_criteria=StoppingCriteriaList(
                [StopOnWords(self.pg.stop_words, self.tokenizer, config.device)]
            ),
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=config.num_beams,
        )

    def predict(self, doc: dict[str, str | list[str]]) -> OWA_PRED:
        prompt = self.pg.generate(self.config.mode, doc)
        generation = self.generator(prompt)[0]["generated_text"]

        # get LAST element between <EVALUATE> tags using regex
        # TODO: write different regex for neurosymbolic mode
        generation = re.findall(
            rf"<EVALUATE>\n*(.+?)\n*<\/EVALUATE>", generation, re.DOTALL
        )[-1]
        generation = generation.strip()
        if self.config.mode == MODEL_MODE.BASELINE:
            return self.evaluate_baseline(generation)
        elif self.config.mode == MODEL_MODE.NEUROSYMBOLIC:
            return self.evaluate_neurosymbolic(generation)


class GeminiModel(BaseModel):
    def __init__(
        self,
        config: GeminiModelConfig,
    ) -> None:
        self.config = config
        genai.configure(api_key=config.google_api_key)
        self.model = genai.GenerativeModel(config.model_name)
        self.pg = config.pg

    def predict(self, doc: dict[str, str | list[str]]) -> OWA_PRED:
        prompt = self.pg.generate(self.config.mode, doc)
        generation = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,  # looks like multiple candidates are not supported
                max_output_tokens=self.config.max_new_tokens,
                # the stop sequences are NOT included in the output
                stop_sequences=["</EVALUATE>"],
            ),
        )
        text = generation.text  # might be different for multiple candidates
        text = text.strip()
        # since the generation stops at </EVALUATE>, and does not include the prompt
        # just generation.text is exactly what we need
        if self.config.mode == MODEL_MODE.BASELINE:
            return self.evaluate_baseline(text)
        elif self.config.mode == MODEL_MODE.NEUROSYMBOLIC:
            return self.evaluate_neurosymbolic(text)


class CohereModel(BaseModel):
    def __init__(
        self,
        config: CohereModelConfig,
    ) -> None:
        self.config = config
        self.model = cohere.Client(api_key=config.api_key)
        self.pg = config.pg

    def predict(
        self, doc: dict[str, str | list[str]], n: int = 1
    ) -> OWA_PRED | list[OWA_PRED]:
        prompt = self.pg.generate(self.config.mode, doc)
        # generate is legacy, cohere is asking us to use chat instead
        # but chat doesn't have a stop_sequences parameter
        # read more here https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat
        generation = self.model.generate(
            prompt,
            end_sequences=["</EVALUATE>"],
            max_tokens=self.config.max_new_tokens,
            num_generations=n,
            model=self.config.model_name,
        )

        if n == 1:
            text = generation[0].text
            text = text.strip()
            if self.config.mode == MODEL_MODE.BASELINE:
                return self.evaluate_baseline(text)
            elif self.config.mode == MODEL_MODE.NEUROSYMBOLIC:
                return self.evaluate_neurosymbolic(text)

        elif n > 1:
            if self.config.mode == MODEL_MODE.BASELINE:
                return [self.evaluate_baseline(g.text.strip()) for g in generation]
            elif self.config.mode == MODEL_MODE.NEUROSYMBOLIC:
                return [self.evaluate_neurosymbolic(g.text.strip()) for g in generation]

        else:
            raise ValueError("n must be a positive integer")


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

    hf_config = HFModelConfig(
        model_name="microsoft/phi-2",
        quantize=True,
        num_beams=5,
    )

    model = HFModel(hf_config)

    print(model.predict(example_doc))

    """ gemini_config = GeminiModelConfig(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    model = GeminiModel(gemini_config)

    print(model.predict(example_doc)) """

    """ cohere_config = CohereModelConfig(
        api_key=os.getenv("COHERE_API_KEY"),
    )

    model = CohereModel(cohere_config)

    print(model.predict(example_doc, n=5)) """
