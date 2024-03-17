import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pprint import pprint, pformat
import glob
import os
from tqdm import tqdm


class DataSample:
    def __init__(
        self,
        id: str,
        filename: str,
        triples: Dict[str, str],
        rules: Dict[str, str],
        abductions: Dict[str, str],
    ) -> None:
        self.id = id
        self.triples = triples
        self.rules = rules
        self.abductions = abductions
        self.filename = filename
        self.answerable = []
        # TODO - parse each triple, rule, and abduction into FOL

        # check if there are any answers, sets to the abduction name
        for k, v in self.abductions.items():
            if "answers" in v and v["answers"] is not None and len(v["answers"]) > 0:
                self.answerable.append(k)

    def __str__(self) -> str:
        return (
            f"DataSample[{self.id}][{self.filename}]\n"
            f" Triples:\n  {pformat(self.triples, indent=1)}\n"
            f" Rules:\n  {pformat(self.rules, indent=1)}\n"
            f" Abductions:\n  {pformat(self.abductions, indent=1)}\n"
            f" Answerable: {self.answerable}"
        )

    def __repr__(self) -> str:
        """DataSample(<id>, <num_answerable>, <num_triples>, <num_rules>, <num_abductions>)

        Returns:
            str: _description_
        """
        return f"DataSample({self.id}, {len(self.answerable)}A, {len(self.triples)}T, {len(self.rules)}R, {len(self.abductions)}AB)"


def get_files(path: str, glob_str: str) -> List[str]:
    """Get all files in a directory matching a glob pattern

    Args:
        path (str): _description_
        glob_str (str): _description_

    Returns:
        List[str]: _description_
    """
    path = os.path.normpath(path)  # format the path
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")
    files = glob.glob(path + glob_str, recursive=True)
    if not files:
        raise ValueError(f"No jsonl files found in {path} using glob {glob_str}")
    return files


def load_multiple_datasets(paths: List[str]) -> Dict[str, List[DataSample]]:
    """Load multiple datasets into a dictionary of lists of DataSamples

    Args:
        paths (List[str]): _description_

    Returns:
        Dict[str, List[DataSample]]: _description_
    """
    samples = {}
    for file in tqdm(
        paths,
        leave=False,
        desc="Loading jsonl files",
        ncols=100,
        unit="file",
    ):
        samples[file] = Dataset(file)
    return samples


class Dataset:

    def __init__(self, path: str) -> None:
        """path should be a directory of directories of jsonl files

        Args:
            path (str): _description_
        """
        self.path = path

        with open(path, "r") as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            data = json.loads(line)
            sample = DataSample(
                id=data["id"],
                filename=path,
                triples=data["triples"] if "triples" in data else {},
                rules=data["rules"] if "rules" in data else {},
                abductions=data["abductions"] if "abductions" in data else {},
            )
            self.samples.append(sample)
        self.samples


if __name__ == "__main__":
    path = "proofwriter-dataset/OWA/depth-2/meta-test.jsonl"
    top_dir = os.path.join("proofwriter-dataset", "OWA")
    paths = get_files(top_dir, "/**/*.jsonl")

    paths = list(
        filter(
            lambda x: os.path.basename(x) == "meta-test.jsonl"
            or os.path.basename(x) == "meta-train.jsonl"
            or os.path.basename(x) == "meta-dev.jsonl",
            paths,
        )
    )
    print(paths)

    # all_datasets = load_multiple_datasets(paths)

    dataset = Dataset(path)
    print(dataset.samples[0])
