import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset

# for language modeling problems how long to use the prefix as
PREFIX_LENGTH: int = 100


@dataclass
class EvaluationExample:
    input: str
    output: str


class DatasetFormat:
    CHAT_FORMAT: str = "chat_format"
    CNN_DM_SUMMARIZATION: str = "cnn_dm_summarization"
    CNN_DM_LM: str = "cnn_dm_lm"
    XSUM_SUMMARIZATION: str = "xsum_summarization"
    HUMAN_EVAL: str = "human_eval"


def LowercaseProcessingFunction(input: str) -> str:
    return input.lower()


# TODO: fix or remove TOPv2 benchmarking
def prepare_evaluation_examples_chat_format(data_path: str) -> List[EvaluationExample]:
    SINGLE_TURN_TEMPLATE: str = "\n[{role}]\n{message}\n[/{role}]"
    evaluation_data_points = []

    def stringify_conversation(conversation: List[Dict[str, str]]) -> str:
        return "".join(
            [
                SINGLE_TURN_TEMPLATE.format(role=x["role"], message=x["message"])
                for x in conversation
            ]
        )

    for line in open(data_path):
        json_line = json.loads(line)
        i: int = 0
        while i < len(json_line["data"]):
            if json_line["data"][i]["role"] == "PARSER":
                evaluation_data_points.append(
                    EvaluationExample(
                        input=stringify_conversation(json_line["data"][1:i])
                        + "\n[PARSER]\n",
                        output=stringify_conversation([json_line["data"][i]]),
                    )
                )
            i += 1
    return evaluation_data_points


def prepare_cnn_dm_lm_format() -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset("cnn_dailymail", "3.0.0")["test"]:
        words = data_point["article"].split()
        evaluation_data_points.append(
            EvaluationExample(
                input=" ".join(words[:PREFIX_LENGTH]),
                output=" ".join(words[PREFIX_LENGTH:]),
            )
        )
    return evaluation_data_points


def prepare_cnn_dm_summarization_format(n_shot: int = 0, seed: int = 42) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        prompt_keys=["article", "highlights"]
        shots = load_dataset("cnn_dailymail", name="3.0.0", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = "Article: " + shots[i][prompt_keys[0]] + "\nSummary: " + shots[i][prompt_keys[1]].replace("\n", "") + "\n"
            prompt_shots += prompt
        prompt_shots += "\n"

    evaluation_data_points = []
    for data_point in load_dataset("cnn_dailymail", name="3.0.0", split="test"):
        article = data_point["article"]
        highlights = data_point["highlights"]
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt_shots + f"Article: {article}\nSummary:",
                output=f" {highlights}",
            )
        )
    return evaluation_data_points

def prepare_xsum_summarization_format(n_shot: int = 0, seed: int = 42) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        prompt_keys=["document", "summary"]
        shots = load_dataset("xsum", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = "Article: " + shots[i][prompt_keys[0]] + "\nSummary: " + shots[i][prompt_keys[1]].replace("\n", "") + "\n"
            prompt_shots += prompt
        prompt_shots += "\n"

    evaluation_data_points = []
    for data_point in load_dataset('xsum', split='test'):
        article = data_point["document"]
        highlights = data_point["summary"]
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt_shots + f"Article: {article}\nSummary:",
                output=f" {highlights}",
            )
        )
    return evaluation_data_points

def prepare_human_eval() -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset('openai_humaneval', split='test'):
        evaluation_data_points.append(
            EvaluationExample(
                input=data_point["prompt"],
                output=data_point["canonical_solution"],
            )
        )
    return evaluation_data_points

def get_data(
    random_shuffle: bool,
    num_samples: int,
    dataset: str,
    data_path: Optional[str] = None,
    n_shot: int = 0,
    seed: int = 42,
) -> List[EvaluationExample]:
    if dataset == DatasetFormat.CHAT_FORMAT:
        evaluation_data_points = prepare_evaluation_examples_chat_format(data_path)
    elif dataset == DatasetFormat.CNN_DM_SUMMARIZATION:
        evaluation_data_points = prepare_cnn_dm_summarization_format(n_shot=n_shot, seed=seed)
    elif dataset == DatasetFormat.XSUM_SUMMARIZATION:
        evaluation_data_points = prepare_xsum_summarization_format(n_shot=n_shot, seed=seed)
    elif dataset == DatasetFormat.CNN_DM_LM:
        evaluation_data_points = prepare_cnn_dm_lm_format()
    elif dataset == DatasetFormat.HUMAN_EVAL:
        evaluation_data_points = prepare_human_eval()
    else:
        raise NotImplementedError(f"Unknown dataset format {dataset}")

    if random_shuffle:
        random.shuffle(evaluation_data_points)

    if num_samples:
        evaluation_data_points = evaluation_data_points[:num_samples]

    return evaluation_data_points
