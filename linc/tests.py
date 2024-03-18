import os
from copy import deepcopy
from time import sleep, time

from custom_types import MODEL_MODE, OWA_PRED
from dataset import *
from datasets import Dataset
from dotenv import load_dotenv
from lm import *
from models import *

load_dotenv()


def test_n_samples(
    model: BaseModel,
    data: Dataset,
    n_samples: int = 10,
    sleep_time: int = 0,
    save: bool = False,
    file: str = "test_results.txt",
) -> tuple[list[OWA_PRED], list[OWA_PRED], str]:
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

    for row in data:
        if "answer" not in row or "question" not in row or "id" not in row:
            print("Skipping row without answer, question, or id")
            continue

        sample_id = row["id"]
        x = {
            "conclusion": row["question"],
            "premises": row["theory"],
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
            f"Sample[{sample_id}]\n Question: {x}\n Answer: {y}"
            f"\n Prediction: {yhat}\n Correct? {correct}\n"
        )
        print(text)
        if save:
            with open(file, "a") as f:
                f.write(text + "\n")
        if sleep_time > 0:
            sleep(sleep_time)

    return seqs["y"], seqs["yhat"], file


def compute_accuracy(y: list[OWA_PRED], yhat: list[OWA_PRED]) -> float:
    """Compute accuracy of the model with y and yhat

    Args:
        y (List[OWA_PRED]): _description_
        yhat (List[OWA_PRED]): _description_

    Returns:
        float: accuracy
    """
    if not y or not yhat or len(y) != len(yhat) or len(y) == 0 or len(yhat) == 0:
        raise ValueError("y and yhat must have the same length and not be empty")
    if not all(isinstance(i, OWA_PRED) for i in y):
        raise ValueError("y must be a list of OWA_PRED")
    if not all(isinstance(i, OWA_PRED) for i in yhat):
        raise ValueError("yhat must be a list of OWA_PRED")
    return sum([1 for i, j in zip(y, yhat) if i == j]) / len(y)


if __name__ == "__main__":
    train, test = get_dataset()
    # random_model = RandomModel()
    geminiconfig = GeminiModelConfig(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    gemini_model = GeminiModel(geminiconfig)
    y, yhat, filename = test_n_samples(
        model=gemini_model,
        data=test,
        n_samples=360,
        sleep_time=5,  # avoid rate limiting
        save=True,
        file=f"output/gemini_test_results_{time()}.txt",
    )
    acc = compute_accuracy(y, yhat)
    print(f"Accuracy: {acc}")
    with open(filename, "a") as f:
        f.write(f"Accuracy: {acc}\n")

    print("Done")
