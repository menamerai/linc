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

        # check if there are any answers, sets to the abduction name
        for k, v in self.abductions.items():
            if "answers" in v and v["answers"] is not None and len(v["answers"]) > 0:
                self.answerable.append(k)

    def __str__(self) -> str:
        return (
            f"DataSample[{self.id}][{self.filename}]\n"
            f" Triples:\n  {pformat(self.triples, indent=1)}\n"
            f" Rules:\n  {pformat(self.rules, indent=1)}\n"
            f" Abductions:\n  {pformat(self.abductions, indent=1)}"
            f" Answerable: {self.answerable}"
        )

    def __repr__(self) -> str:
        """DataSample(<id>, <num_answerable>, <num_triples>, <num_rules>, <num_abductions>)

        Returns:
            str: _description_
        """
        return f"DataSample({self.id}, {len(self.answerable)}A, {len(self.triples)}T, {len(self.rules)}R, {len(self.abductions)}AB)"


class Dataset:

    def __init__(self, path: str, glob_str: str = "/**/*.jsonl") -> None:
        """path should be a directory of directories of jsonl files

        Args:
            path (str): _description_
        """
        self.path = path
        self.glob_str = glob_str

        # load all jsonl files
        path = os.path.normpath(path)  # format the path
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")
        self.files = glob.glob(path + glob_str, recursive=True)
        if not self.files:
            raise ValueError(f"No jsonl files found in {path} using glob {glob_str}")

        # load all jsonl files, setup like the directory structure
        self.samples = {}
        for file in tqdm(
            self.files,
            leave=False,
            desc="Loading jsonl files",
            ncols=100,
            unit="file",
        ):
            samples = Dataset.load_jsonl(file)
            self.samples[file] = samples

    @classmethod
    def load_jsonl(cls, path: str) -> List[DataSample]:
        """Load a jsonl file into a list of DataSamples

        Args:
            path (str): _description_

        Returns:
            List[DataSample]: _description_
        """
        with open(path, "r") as f:
            lines = f.readlines()
        samples = []
        for line in lines:
            data = json.loads(line)
            sample = DataSample(
                id=data["id"],
                filename=path,
                triples=data["triples"] if "triples" in data else {},
                rules=data["rules"] if "rules" in data else {},
                abductions=data["abductions"] if "abductions" in data else {},
            )
            samples.append(sample)
        return samples


if __name__ == "__main__":
    path = "proofwriter-dataset/OWA/depth-2/meta-abduct-test.jsonl"
    # dataset = Dataset.load_jsonl(path)
    # pprint(dataset[0])

    dataset = Dataset(os.path.join("proofwriter-dataset", "OWA"))
    print(dataset.samples.keys())
