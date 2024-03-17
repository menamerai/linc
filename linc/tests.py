from models import *
from dataset import *
from custom_types import OWA_PRED, MODEL_MODE
from copy import deepcopy

# implement ProofWriter ( download script) and test against RandomModel
# add subsample method that samples a random subset of the data (in the paper it's 360 samples)


def test_n_samples(model: BaseModel, data, n_samples: int = 10) -> float:
    assert len(data) >= n_samples, f"dataset too small for {n_samples} samples"
    data = deepcopy(data)
    data.shuffle()
    data = data.select(range(n_samples))

    num_correct = 0

    for row in data:
        sample_id = row["id"]
        x = row["question"]
        y = row["answer"]
        # parse answer True, False, or Uncertain into OWA_PRED
        y = model.evaluate_baseline(y)
        yhat = model.predict(x)

        correct = y == yhat
        if correct:
            num_correct += 1

        text = (
            f"Sample[{sample_id}]\n Question: {x}\n Answer: {y}"
            f"\n Prediction: {yhat}\n Correct? {correct}\n"
        )
        print(text)

    total_accuracy = num_correct / n_samples
    print(f"Total accuracy: {total_accuracy}")
    return total_accuracy


if __name__ == "__main__":
    train, test = get_dataset()
    random_model = RandomModel()
    test_n_samples(random_model, test, 10)
