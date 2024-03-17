from models import *
from datasets import *
from custom_types import OWA_PRED, MODEL_MODE


# implement ProofWriter ( download script) and test against RandomModel
# add subsample method that samples a random subset of the data (in the paper it's 360 samples)


def test_n_samples(model: BaseModel, data, n_samples: int = 10) -> None:
    assert len(data) >= n_samples, f"dataset too small for {n_samples} samples"
    data = data.sample(n_samples)
    for i, row in data.iterrows():
        pred = model.predict(row["question"])
        print(f"Question: {row['question']}")
        print(f"Answer: {row['answer']}")
        print(f"Prediction: {pred}\n")


if __name__ == "__main__":
    train, test = get_dataset()
    random_model = RandomModel()
    test_n_samples(random_model, test, 10)
