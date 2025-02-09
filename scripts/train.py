import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from torch.utils.data import DataLoader

from lettucedetect.models.trainer import Trainer
from lettucedetect.datasets.ragtruth import RagTruthDataset
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthData


def main():
    data_path = Path("data/ragtruth/ragtruth_data.json")
    rag_truth_data = RagTruthData.from_json(json.loads(data_path.read_text()))

    train_samples = [
        sample for sample in rag_truth_data.samples if sample.split == "train"
    ]
    test_samples = [
        sample for sample in rag_truth_data.samples if sample.split == "test"
    ]

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )

    train_dataset = RagTruthDataset(train_samples, tokenizer)
    test_dataset = RagTruthDataset(test_samples, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, collate_fn=data_collator
    )

    model = AutoModelForTokenClassification.from_pretrained(
        "answerdotai/ModernBERT-base", num_labels=2
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=6,
        learning_rate=1e-5,
        save_path="output/hallucination_detector",
    )

    trainer.train()


if __name__ == "__main__":
    main()
