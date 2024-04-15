import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from custom_types import MODEL_MODE
from transformers import AutoTokenizer, BitsAndBytesConfig, StoppingCriteria
from utils import bnb_factory, convert_to_nltk_rep


class PromptGenerator:
    """train_dataset = "minimario/FOLIO" """

    def __init__(self, n: int = 3):
        self.common_instructions = (
            "The following is a first-order logic (FOL) problem.\n"
        )
        self.common_instructions += "The problem is to determine whether the conclusion follows from the premises.\n"
        self.common_instructions += "The premises are given in the form of a set of first-order logic sentences.\n"
        self.common_instructions += "The conclusion is given in the form of a single first-order logic sentence.\n"
        self.n_shots = n
        self.stop_words = ["</EVALUATE>"]
        """ self.dataset = load_dataset(self.train_dataset, split="train")
        self.n_indices = [23, 60, 125]  # true, false, uncertain """

    def generate(self, mode: MODEL_MODE, doc: dict[str, str | list[str]], reason: bool = False) -> str:
        # instructions
        prompt = self.common_instructions
        if mode == MODEL_MODE.BASELINE:
            prompt += "The task is to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises.\n\n"
        elif mode == MODEL_MODE.NEUROSYMBOLIC:
            prompt += "The task is to translate each of the premises and conclusions into FOL expressions, "
            prompt += "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises."
            prompt += "Expressions should be adhere to the format of the Python NLTK package logic module. Do NOT use special characters like ∃ or ∀.\n"
        else:
            raise ValueError(f"Invalid mode: {mode}, expected one of {self.modes}")
         
        if reason:
            prompt += "MANDATORY RULE: FOL expressions CANNOT use the same names for variables and functions. Functions are followed by parenthesis (e.g. Ohio(...)), and variables are not.\n"
            prompt += "Use the <REASONING> tags to think step by step and add additional non-reflexive premises to prove/disprove the conclusion. Do NOT output reflexive premises (e.g. all cars are cars) as new premises during reasoning.\n"

        prompt += "ONLY OUTPUT THE GENERATIONS SURROUNDED BY THE <EVALUATE> TAGS, DO NOT OUTPUT EXTRA TEXT. Take a deep breath and get started.\n\n"
        # get examples
        if self.n_shots > 0:
            prompt += self.get_and_format_example(mode, reason=reason)

        # write frame for problem
        prompt += "<PREMISES>\n"
        for premise in doc["premises"]:
            prompt += premise + "\n"
        prompt += "</PREMISES>\n"
        prompt += "<CONCLUSION>\n"
        prompt += doc["conclusion"] + "\n"
        prompt += "</CONCLUSION>\n"
        if reason:
            prompt += "<REASONING>\n"
        else:
            prompt += "<EVALUATE>\n"

        return prompt

    def get_and_format_example(self, mode: MODEL_MODE, reason: bool = False) -> str:
        examples = self.get_examples(mode, self.n_shots)
        formatted_examples = ""
        for example in examples:
            formatted_examples += "<PREMISES>\n"
            for premise in example["premises"]:
                formatted_examples += premise + "\n"
            formatted_examples += "</PREMISES>\n"
            formatted_examples += "<CONCLUSION>\n"
            formatted_examples += example["conclusion"] + "\n"
            formatted_examples += "</CONCLUSION>\n"
            if reason:
                formatted_examples += "<REASONING>\n"
                formatted_examples += example["reasoning"] + "\n"
                formatted_examples += "</REASONING>\n"
            formatted_examples += "<EVALUATE>\n"
            if mode == MODEL_MODE.BASELINE:
                formatted_examples += example["label"] + "\n"
            elif mode == MODEL_MODE.NEUROSYMBOLIC:
                for premise, fol in zip(example["premises"], example["premises_FOL"]):
                    formatted_examples += f"TEXT: {premise}\n"
                    formatted_examples += f"FOL: {fol}\n"

                formatted_examples += f"CONCLUSION: {example['conclusion']}\n"
                formatted_examples += f"FOL: {example['conclusion_FOL']}\n"
            formatted_examples += f"</EVALUATE>\n\n"

        return formatted_examples

    def get_examples(
        self, mode: MODEL_MODE, n: int
    ) -> list[dict[str, str | list[str]]]:
        # let's make a static dict of examples for now
        assert n <= 3, "Only 3 examples available"
        examples = [
            {
                "premises": [
                    "A La Liga soccer team ranks higher than another if it receives more points.",
                    "If two La Liga soccer teams recieve the same points, the team which recieves more points from the games between the two teams ranks higher.",
                    "Real Madrid and Barcelona are both La Liga soccer teams.",
                    "In La Liga 2021-2022, Real Madrid recieves 86 points and Barcelon recieves 73 points.",
                    "In La Liga 2021-2022, Real Madrid and Barcelona both recieve 3 points from the games between them.",
                ],
                "conclusion": "In La Liga 2021-2022, Real Madrid ranks higher than Barcelona.",
                "conclusion_FOL": "HigherRank(RealMadrid, Barcelona)",
                "label": "True",
                "reasoning": "Let's think step by step. We know that a soccer team is a type of team, so we can add a premise that all La Liga soccer teams are also soccer teams. This is the only premise we can reasonably add, leaving us with our final premises:\nPREMISE: A La Liga soccer team ranks higher than another if it receives more points.\nPREMISE: If two La Liga soccer teams recieve the same points, the team which recieves more points from the games between the two teams ranks higher.\nPREMISE: Real Madrid and Barcelona are both La Liga soccer teams.\nPREMISE: In La Liga 2021-2022, Real Madrid recieves 86 points and Barcelona recieves 73 points.\nPREMISE: In La Liga 2021-2022, Real Madrid and Barcelona both recieve 3 points from the games between them.\nPREMISE: All La Liga soccer teams must also be soccer teams.",
                "premises_FOL": [
                    "∀x ∀y (LaLiga(x) ∧ LaLiga(y) ∧ MorePoints(x, y) → HigherRank(x, y))",
                    "∀x ∀y (LaLiga(x) ∧ LaLiga(y) ∧ ¬MorePoints(x, y) ∧ ¬MorePoints(y, x) ∧ MorePointsInGameBetween(x, y) → HigherRank(x, y))",
                    "LaLiga(realMadrid) ∧ LaLiga(barcelona)",
                    "MorePoints(realMadrid, barcelona)",
                    "¬MorePointsInGameBetween(realMadrid, barcelona) ∧ ¬MorePointsInGameBetween(barcelona, realMadrid)",
                ],
            },
            {
                "premises": [
                    "All athletes are good at sports.",
                    "All Olympic gold medal winners are good athletes.",
                    "No scientists are good at sports.",
                    "All Nobel laureates are scientists.",
                    "Amy is good at sports or Amy is an Olympic gold medal winner.",
                    "If Amy is not a Nobel laureate, then Amy is not an Olympic gold medal winner.",
                ],
                "conclusion": "If Amy is not an Olympic gold medal winner, then Amy is a Nobel laureate.",
                "conclusion_FOL": "-OlympicGoldMedalWinner(Amy) -> NobelLaureate(Amy)",
                "label": "False",
                "reasoning": "Let's think step by step. We know that a good athlete must also be an athlete, so we can add that to our list of premises. We also know that all Nobel laureates are good at science. This leaves us with our final premises:\nPREMISE: All athletes are good at sports.\nPREMISE: All Olympic gold medal winners are good athletes.\nPREMISE: No scientists are good at sports.\nPREMISE: All Nobel laureates are scientists.\nPREMISE: Amy is good at sports or Amy is an Olympic gold medal winner.\nPREMISE: If Amy is not a Nobel laureate, then Amy is not an Olympic gold medal winner.\nPREMISE: All good athletes are athletes.\nPREMISE: All Nobel laureates are good at science.",
                "premises_FOL": [
                    "∀x (Athlete(x) → GoodAtSports(x))",
                    "∀x (OlympicGoldMedalWinner(x) → Athlete(x))",
                    "∀x (Scientist(x) → ¬GoodAtSports(x))",
                    "∀x (NobelLaureate(x) → Scientist(x))",
                    "GoodAtSports(amy) ∨ OlympicGoldMedalWinner(amy)",
                    "¬NobelLaureate(amy) → ¬OlympicGoldMedalWinner(amy)",
                ],
            },
            {
                "premises": [
                    "All dispensable things are environment-friendly. ",
                    "All woodware is dispensable.",
                    "All paper is woodware. ",
                    "No good things are bad. ",
                    "All environment-friendly things are good.",
                    "A worksheet is either paper or is environment-friendly.",
                ],
                "conclusion": "A worksheet is not dispensable.",
                "conclusion_FOL": "-Dispensable(worksheet)",
                "label": "Uncertain",
                "reasoning": "Let's think step by step. Looking through our premises, we see that there are no ambiguous phrases that necessitate extra premises. Therefore, our premises are unchanged, leaving us with our final premises:\nPREMISE: All dispensable things are environment-friendly.\nPREMISE: All woodware is dispensable.\nPREMISE: All paper is woodware.\nPREMISE: No good things are bad.\nPREMISE: All environment-friendly things are good.\nPREMISE: A worksheet is either paper or is environment-friendly.",
                "premises_FOL": [
                    "∀x (Dispensable(x) → EnvironmentFriendly(x))",
                    "∀x (Woodware(x) → Dispensable(x))",
                    "∀x (Paper(x) → Woodware(x))",
                    "∀x (Good(x) → ¬Bad(x))",
                    "∀x (EnvironmentFriendly(x) → Good(x))",
                    "Paper(worksheet) ⊕ EnvironmentFriendly(worksheet)",
                ],
            },
        ]

        if mode == MODEL_MODE.NEUROSYMBOLIC:
            for example in examples:
                example["premises_FOL"] = [
                    convert_to_nltk_rep(premise) for premise in example["premises_FOL"]
                ]
                example["conclusion_FOL"] = convert_to_nltk_rep(
                    example["conclusion_FOL"]
                )
        rel = examples[:n]
        random.shuffle(rel)
        return rel

    def __call__(self, mode: MODEL_MODE, s: str) -> str:
        return self.generate(mode, s)


class StopOnWords(StoppingCriteria):
    # stopper if the model generated <\EVALUATE>
    def __init__(
        self, stop_words: list[str], tokenizer: AutoTokenizer, device: str = "cpu"
    ):
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.device = device

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        stop_token_ids = [
            self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt")[
                0
            ].to(self.device)
            for word in self.stop_words
        ]
        for stop_token_id in stop_token_ids:
            # check last n tokens with n = len(token_id)
            # print(f"test {self.tokenizer.batch_decode(input_ids[0][-len(stop_token_id):])} against {self.tokenizer.batch_decode(stop_token_ids)}")
            if torch.equal(input_ids[0][-len(stop_token_id) :], stop_token_id):
                return True
        return False


@dataclass
class HFModelConfig:
    """
    Arguments for HF model configuration
    """

    model_name: str = field(
        default="microsoft/phi-2",
        metadata={"help": "The model name of the transformer model on HF hub"},
    )
    pg: PromptGenerator = field(
        default=PromptGenerator(),
        metadata={"help": "Prompt generator for the model"},
    )
    q_config: Optional[BitsAndBytesConfig] = field(
        default_factory=bnb_factory,
        metadata={"help": "Quantization configuration for the model"},
    )
    mode: MODEL_MODE = field(
        default=MODEL_MODE.BASELINE,
        metadata={"help": "Mode of the model, between baseline and neurosymbolic"},
    )
    device: torch.device = field(
        default=(
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda")
        ),
        metadata={"help": "Device to run the model on"},
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Maximum length of the generated text (prompt included)"},
    )
    num_beams: int = field(
        default=1,  # 1 means greedy search
        metadata={"help": "Number of beams for the generation, used for beam search"},
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling for the generation"},
    )
    temperature: float = field(
        default=0.8,
        metadata={"help": "Sampling temperature for the generation"},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top p value for nucleus sampling"},
    )


@dataclass
class GeminiModelConfig:
    """
    Arguments for Gemini model configuration
    """

    google_api_key: str = field(
        metadata={"help": "Google API key for the Gemini model"},
    )
    pg: PromptGenerator = field(
        default=PromptGenerator(),
        metadata={"help": "Prompt generator for the model"},
    )
    mode: MODEL_MODE = field(
        default=MODEL_MODE.BASELINE,
        metadata={"help": "Mode of the model, between baseline and neurosymbolic"},
    )
    model_name: str = field(
        default="gemini-pro",
        metadata={"help": "The model name of the Gemini model"},
    )
    max_new_tokens: int = field(
        default=1000,
        metadata={"help": "Maximum number of new tokens to generate"},
    )
    temperature: float = field(
        default=0.8,
        metadata={"help": "Sampling temperature for the generation"},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top p value for nucleus sampling"},
    )


@dataclass
class CohereModelConfig:
    """
    Arguments for Cohere model configuration
    """

    api_key: str = field(
        metadata={"help": "API key for the Cohere model"},
    )
    pg: PromptGenerator = field(
        default=PromptGenerator(),
        metadata={"help": "Prompt generator for the model"},
    )
    num_generations: int = field(
        default=5,
        metadata={"help": "The number of generations to run"}
    )
    mode: MODEL_MODE = field(
        default=MODEL_MODE.BASELINE,
        metadata={"help": "Mode of the model, between baseline and neurosymbolic"},
    )
    reason: bool = field(
        default=False,
        metadata={"help": "Whether to use chain-of-thought reasoning to add new premises"}
    )
    model_name: str = field(
        default="command",
        metadata={"help": "The model name of the Cohere model"},
    )
    max_new_tokens: int = field(
        default=1000,
        metadata={"help": "Maximum number of new tokens to generate"},
    )
    temperature: float = field(
        default=0.8,
        metadata={"help": "Sampling temperature for the generation"},
    )


if __name__ == "__main__":
    prompt_gen = PromptGenerator()
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
    prompt = prompt_gen.generate(MODEL_MODE.NEUROSYMBOLIC, example_doc, reason=True)
    print(prompt)
