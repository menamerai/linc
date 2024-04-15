import argparse
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BitsAndBytesConfig

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
        # make directory if it doesn't exist
        os.makedirs(os.path.dirname(file), exist_ok=True)

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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_type", type=str, default="hf")
    # default model of gemini is gemini-prod, cohere is command
    args.add_argument("--model_name", type=str, default="bigcode/starcoderplus")
    # args.add_argument("--quantize", action="store_true")
    args.add_argument("--q_4bit", action="store_true")
    args.add_argument("--q_8bit", action="store_true")
    args.add_argument("--num_beams", type=int, default=5)
    args.add_argument("--mode", type=str, default="neurosymbolic")
    args.add_argument("--sleep_time", type=int, default=0)
    args.add_argument("--filename_suffix", type=str, default="")
    args.add_argument("--do_sample", action="store_true")
    args.add_argument("--top_p", type=int, default=0.95)
    args.add_argument("--temperature", type=float, default=0.8)
    args = args.parse_args()

    train, test = get_dataset()

    if args.mode == "neurosymbolic":
        mode = MODEL_MODE.NEUROSYMBOLIC
    elif args.mode == "baseline":
        mode = MODEL_MODE.BASELINE
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # check if model is a hf model or gemini/cohere model
    if args.model_type == "hf":
        if args.q_4bit:
            print("Using 4-bit quantization")
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif args.q_8bit:
            print("Using 8-bit quantization")
            q_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            print("Not quantizing model")
            q_config = BitsAndBytesConfig()
        if args.do_sample:
            hf_config = HFModelConfig(
                model_name=args.model_name,
                num_beams=args.num_beams,
                mode=mode,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                q_config=q_config,
            )
        else:
            hf_config = HFModelConfig(
                model_name=args.model_name,
                num_beams=args.num_beams,
                mode=mode,
                q_config=q_config,
            )
        model = HFModel(hf_config)
    elif args.model_type == "gemini":
        gemini_config = GeminiModelConfig(
            model_name=args.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            mode=mode,
            max_new_tokens=4096,
            temperature=args.temperature,
        )
        model = GeminiModel(gemini_config)
    elif args.model_type == "cohere":
        cohere_config = CohereModelConfig(
            model_name=args.model_name,
            api_key=os.getenv("COHERE_API_KEY"),
            mode=mode,
            max_new_tokens=4096,
            temperature=args.temperature,
        )
        model = CohereModel(cohere_config)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    os.makedirs("output", exist_ok=True)

    if args.model_type == "hf":
        # grab latter part of model_name
        model_name = args.model_name.split("/")[-1] + "_" + args.filename_suffix
    else:
        model_name = args.model_name + "_" + args.filename_suffix

    y, yhat, filename = test_n_samples(
        model, test, 360, sleep_time=args.sleep_time, file=model_name
    )
    y = [i.value for i in y]
    yhat = [i.value for i in yhat]
    acc = accuracy_score(y, yhat)
    cf = confusion_matrix(y, yhat)
    report = classification_report(y, yhat)
    print(f"Accuracy:\n\t{acc}")
    print(f"Confusion Matrix:\n{cf}")
    print(f"Classification Report:\n{report}")

    if filename:
        with open(filename, "a") as f:
            f.write(f"Final Results\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Confusion Matrix: {cf}\n")
            f.write(f"Classification Report: {report}\n")
