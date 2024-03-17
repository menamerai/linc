from enum import Enum

from datasets import load_dataset
from utils import convert_to_nltk_rep

MODEL_MODE = Enum("MODEL_MODE", ["BASELINE", "NEUROSYMBOLIC"])


class PromptGenerator:
    """train_dataset = "minimario/FOLIO" """

    container = ("<EVALUATE>", "</EVALUATE>")

    def __init__(self, n: int = 3):
        self.common_instructions = """
        The following is a first-order logic (FOL) problem.
        The problem is to determine whether the conclusion follows from the premises.
        The premises are given in the form of a set of first-order logic sentences.
        The conclusion is given in the form of a single first-order logic sentence.
        """
        self.n_shots = n
        """ self.dataset = load_dataset(self.train_dataset, split="train")
        self.n_indices = [23, 60, 125]  # true, false, uncertain """

    def generate(self, mode: MODEL_MODE, doc: dict[str, str | list[str]]) -> str:
        # instructions
        prompt = self.common_instructions
        if mode == MODEL_MODE.BASELINE:
            prompt += "The task is to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises.\n\n"
        elif mode == MODEL_MODE.NEUROSYMBOLIC:
            prompt += "The task is to translate each of the premises and conclusions into FOL expressions, "
            prompt += "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises."
            prompt += "Expressions should be adhere to the format of the Python NLTK package logic module.\n\n"
        else:
            raise ValueError(f"Invalid mode: {mode}, expected one of {self.modes}")

        # get examples
        if self.n_shots > 0 and mode == MODEL_MODE.BASELINE:
            prompt += self.get_and_format_example(mode)

        # write frame for problem
        prompt += "<PREMISES>\n"
        for premise in doc["premises"]:
            prompt += premise + "\n"
        prompt += "</PREMISES>\n"
        prompt += "<CONCLUSION>\n"
        prompt += doc["conclusion"] + "\n"
        prompt += "</CONCLUSION>\n"
        prompt += "<EVALUATE>\n"

        return prompt

    def get_and_format_example(self, mode: MODEL_MODE) -> str:
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
            formatted_examples += "<EVALUATE>\n"
            if mode == MODEL_MODE.BASELINE:
                formatted_examples += example["label"] + "\n"
            elif mode == MODEL_MODE.NEUROSYMBOLIC:
                for premise, fol in zip(examples["premises"], examples["premises_FOL"]):
                    formatted_examples += f"TEXT: {premise}\n"
                    formatted_examples += f"FOL: {fol}\n"
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
                "label": "True",
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
                "label": "False",
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
                "conclusion": "A worksheet is dispensable.",
                "label": "Uncertain",
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
                    convert_to_nltk_rep(premise) for premise in example["premises"]
                ]

        return examples[:n]

    def __call__(self, mode: MODEL_MODE, s: str) -> str:
        return self.generate(mode, s)


if __name__ == "__main__":
    prompt_gen = PromptGenerator()
    prompt = prompt_gen.get_examples(MODEL_MODE.NEUROSYMBOLIC, 3)
    print(prompt)
