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
    evaluate_detector_char_level,
    evaluate_model,
    evaluate_model_example_level,
    print_metrics,
)
from lettucedetect.models.inference import HallucinationDetector
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
        "--evaluation_type",
        type=str,
        default="example_level",
        help="Evaluation type (token_level, example_level or char_level)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    rag_truth_data = RagTruthData.from_json(json.loads(data_path.read_text()))

    # Filter test samples from the data
    test_samples = [
        sample for sample in rag_truth_data.samples if sample.split == "test"
    ]

    print(f"\nEvaluating model on test samples: {len(test_samples)}")

    if args.evaluation_type in {"token_level", "example_level"}:
        # Initialize model and tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(
            device
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # Prepare the dataset and dataloader
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

        eval_map = {
            "token_level": (evaluate_model, "Token-Level Evaluation"),
            "example_level": (evaluate_model_example_level, "Example-Level Evaluation"),
        }
        eval_fn, eval_title = eval_map[args.evaluation_type]
        print(f"\n---- {eval_title} ----")
        metrics = eval_fn(model, test_loader, device)
        print_metrics(metrics)

    elif args.evaluation_type == "char_level":
        # Character-level span evaluation using the detector
        detector = HallucinationDetector(
            method="transformer", model_path=args.model_path
        )

        print("\n---- Character-Level Span Evaluation ----")
        char_metrics = evaluate_detector_char_level(detector, test_samples)
        print(f"  Precision: {char_metrics['precision']:.4f}")
        print(f"  Recall: {char_metrics['recall']:.4f}")
        print(f"  F1: {char_metrics['f1']:.4f}")
    else:
        raise ValueError(f"Invalid evaluation type: {args.evaluation_type}")


if __name__ == "__main__":
    main()
