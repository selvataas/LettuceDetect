from datasets import load_dataset

dataset = load_dataset("wandb/RAGTruth-processed", split="train")
dataset.to_json("data/ragtruth/response.jsonl", lines=True, force_ascii=False)

print(dataset[0])