import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support, classification_report
from typing import Dict, Any
from tqdm.auto import tqdm
from pathlib import Path
import json
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthData
from lettucedetect.model.dataset import RagTruthDataset


def evaluate_model(
    model: AutoModelForTokenClassification,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a model for hallucination detection.

    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        verbose: Whether to print detailed metrics

    Returns:
        Dictionary containing metrics including:
        - precision, recall, f1 for hallucination detection
        - detailed classification report if verbose=True
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Disable gradients for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Only evaluate on tokens that have labels (not -100)
            mask = batch["labels"] != -100
            predictions = predictions[mask].cpu().numpy()
            labels = batch["labels"][mask].cpu().numpy()

            all_preds.extend(predictions)
            all_labels.extend(labels)

    # Calculate metrics for both classes
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[0, 1],  # Calculate for both classes
        average=None,
    )

    results = {
        "supported": {  # Class 0
            "precision": precision[0],
            "recall": recall[0],
            "f1": f1[0],
        },
        "hallucinated": {  # Class 1
            "precision": precision[1],
            "recall": recall[1],
            "f1": f1[1],
        },
    }

    if verbose:
        # Add detailed classification report
        report = classification_report(
            all_labels, all_preds, target_names=["Supported", "Hallucinated"], digits=4
        )
        print("\nDetailed Classification Report:")
        print(report)
        results["classification_report"] = report

    return results


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print evaluation metrics in a readable format.

    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print("\nEvaluation Results:")
    print("\nHallucination Detection (Class 1):")
    print(f"  Precision: {metrics['hallucinated']['precision']:.4f}")
    print(f"  Recall: {metrics['hallucinated']['recall']:.4f}")
    print(f"  F1: {metrics['hallucinated']['f1']:.4f}")

    print("\nSupported Content (Class 0):")
    print(f"  Precision: {metrics['supported']['precision']:.4f}")
    print(f"  Recall: {metrics['supported']['recall']:.4f}")
    print(f"  F1: {metrics['supported']['f1']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a hallucination detection model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the evaluation data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    args = parser.parse_args()

    # Load the full model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and prepare evaluation data
    data_path = Path(args.data_path)
    rag_truth_data = RagTruthData.from_json(json.loads(data_path.read_text()))

    # Get test split
    test_samples = [
        sample for sample in rag_truth_data.samples if sample.split == "test"
    ]

    # Create dataset and dataloader
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

    # Run evaluation
    metrics = evaluate_model(model, test_loader, device, verbose=True)
    print_metrics(metrics)
