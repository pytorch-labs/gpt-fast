import json
import scripts.settings as settings

def parse_prompts_from_HFdatasets():
    from datasets import load_dataset

    dataset = load_dataset("openai_humaneval", split='test')

    for counter in range(len(dataset)):
        settings.texts.append(dataset[counter]['prompt'])