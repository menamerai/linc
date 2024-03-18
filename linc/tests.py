import os
from copy import deepcopy
from datetime import datetime as dt
from time import sleep

from custom_types import OWA_PRED
from dataset import *
from datasets import Dataset
from dotenv import load_dotenv
from lm import *
from models import *
from sklearn.metrics import accuracy_score, confusion_matrix

load_dotenv()


def test_n_samples(
    model: BaseModel,
    data: Dataset,
    n_samples: int = 10,
    file: str | None = None,
    sleep_time: int = 5,
) -> tuple[list[OWA_PRED], list[OWA_PRED], str | None]:
    """Test the model on n_samples from the dataset

    Args:
        model (BaseModel): model
        data (_type_): HuggingFace dataset
        n_samples (int, optional): how many samples to test. Defaults to 10.

    Returns:
        Union[List[OWA_PRED], List[OWA_PRED]]: List of Y and list Yhat
    """
    assert (
        len(data) >= n_samples
    ), f"dataset length ({len(data)} too small for {n_samples} samples"
    data = deepcopy(data)
    data.shuffle()
    data = data.select(range(n_samples))

    seqs = {"x": [], "y": [], "yhat": []}

    if file:
        file = (
            f"output/{file}_s{n_samples}_{dt.now().strftime('%d-%m-%Y_%H-%M-%S')}.txt"
        )

    for row in data:
        if "answer" not in row or "question" not in row or "id" not in row:
            print("Skipping row without answer, question, or id")
            continue

        sample_id = row["id"]
        x = {
            "conclusion": row["question"],
            "premises": [
                f"{s.strip()}." for s in row["theory"].split(".") if len(s.strip()) > 0
            ],  # split theory into premises at "."
        }
        y = row["answer"]
        # parse answer True, False, or Uncertain into OWA_PRED
        y = model.evaluate_baseline(y)
        yhat = model.predict(x)
        seqs["x"].append(x)
        seqs["y"].append(y)
        seqs["yhat"].append(yhat)

        correct = y == yhat
        text = (
            f"Sample[{sample_id}]\n Question: {x['conclusion']}\n Premises: {x['premises']}\n Answer: {y}"
            f"\n Prediction: {yhat}\n Correct? {correct}\n"
        )
        print(text)
        if file:
            with open(file, "a") as f:
                f.write(text + "\n")

        sleep(sleep_time)

    return seqs["y"], seqs["yhat"], file


# def compute_accuracy(y: List[OWA_PRED], yhat: List[OWA_PRED]) -> float:
#     """Compute accuracy of the model with y and yhat

#     Args:
#         y (List[OWA_PRED]): _description_
#         yhat (List[OWA_PRED]): _description_

#     Returns:
#         float: accuracy
#     """
#     if not y or not yhat or len(y) != len(yhat) or len(y) == 0 or len(yhat) == 0:
#         raise ValueError("y and yhat must have the same length and not be empty")
#     if not all(isinstance(i, OWA_PRED) for i in y):
#         raise ValueError("y must be a list of OWA_PRED")
#     if not all(isinstance(i, OWA_PRED) for i in yhat):
#         raise ValueError("yhat must be a list of OWA_PRED")
#     return sum([1 for i, j in zip(y, yhat) if i == j]) / len(y)


if __name__ == "__main__":
    train, test = get_dataset()
    gemini_config = GeminiModelConfig(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    gemini_model = GeminiModel(gemini_config)
    y, yhat, filename = test_n_samples(
        gemini_model, test, 360, sleep_time=5, file="gemini"
    )
    y = [i.value for i in y]
    yhat = [i.value for i in yhat]
    acc = accuracy_score(y, yhat)
    cf = confusion_matrix(y, yhat)
    print(f"Accuracy:\n\t{acc}")
    print(f"Confusion Matrix:\n{cf}")

    if filename:
        with open(filename, "a") as f:
            f.write(f"Final Results\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Confusion Matrix: {cf}\n")
