from datasets import load_dataset

DATASET_URL = "theoxo/proofwriter-deduction-balanced"


def get_dataset() -> list:
    """Load the dataset
    Dataset({
        features: ['id', 'theory', 'question', 'answer', 'QDep'],
        num_rows: 360
    })
    Returns:
        list: _description_
    """
    train = load_dataset(DATASET_URL, split="train")
    test = load_dataset(DATASET_URL, split="test")
    return train, test


if __name__ == "__main__":
    train, test = get_dataset()
    print(train)
    print(test)
