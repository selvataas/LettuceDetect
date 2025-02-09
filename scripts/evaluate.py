import json
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from torch.utils.data import DataLoader
from lettucedetect.models.evaluator import (
    evaluate_model,
    evaluate_model_example_level,
    print_metrics,
    print_example_level_metrics,
)
from lettucedetect.datasets.ragtruth import RagTruthDataset
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthData
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a hallucination detection model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the evaluation data (JSON format)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    args = parser.parse_args()

    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_path = Path(args.data_path)
    rag_truth_data = RagTruthData.from_json(json.loads(data_path.read_text()))

    test_samples = [
        sample for sample in rag_truth_data.samples if sample.split == "test"
    ]

    test_dataset = RagTruthDataset(test_samples, tokenizer)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    print(f"\nEvaluating model on {device}")
    print(f"Test samples: {len(test_dataset)}\n")

    # Token-level evaluation
    print("---- Token-Level Evaluation ----")
    token_metrics = evaluate_model(model, test_loader, device, verbose=True)
    print_metrics(token_metrics)

    # Example-level evaluation
    print("---- Example-Level Evaluation ----")
    example_metrics = evaluate_model_example_level(
        model, test_loader, device, verbose=True
    )
    print_example_level_metrics(example_metrics)


if __name__ == "__main__":
    main()
