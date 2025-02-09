import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm.auto import tqdm


def evaluate_model(
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate a model for hallucination detection.

    Returns:
        Dict containing metrics for both classes:
        {
            "supported": {"precision": float, "recall": float, "f1": float},
            "hallucinated": {"precision": float, "recall": float, "f1": float}
        }
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits: torch.Tensor = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Only evaluate on tokens that have labels (not -100)
            mask = batch["labels"] != -100
            predictions = predictions[mask].cpu().numpy()
            labels = batch["labels"][mask].cpu().numpy()

            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], average=None
    )

    results: dict[str, dict[str, float]] = {
        "supported": {  # Class 0
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        },
        "hallucinated": {  # Class 1
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
        },
    }

    if verbose:
        report = classification_report(
            all_labels, all_preds, target_names=["Supported", "Hallucinated"], digits=4
        )
        print("\nDetailed Classification Report:")
        print(report)
        results["classification_report"] = report

    return results


def print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Print evaluation metrics in a readable format."""
    print("\nEvaluation Results:")
    print("\nHallucination Detection (Class 1):")
    print(f"  Precision: {metrics['hallucinated']['precision']:.4f}")
    print(f"  Recall: {metrics['hallucinated']['recall']:.4f}")
    print(f"  F1: {metrics['hallucinated']['f1']:.4f}")

    print("\nSupported Content (Class 0):")
    print(f"  Precision: {metrics['supported']['precision']:.4f}")
    print(f"  Recall: {metrics['supported']['recall']:.4f}")
    print(f"  F1: {metrics['supported']['f1']:.4f}")


def evaluate_model_example_level(
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Evaluate a model for hallucination detection at the example level.

    For each example, if any token is marked as hallucinated (label=1),
    then the whole example is considered hallucinated. Otherwise, it is supported.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader providing the evaluation batches.
        device: Device on which to perform evaluation.
        verbose: If True, prints a detailed classification report.

    Returns:
        A dict containing example-level metrics:
        {
            "supported": {"precision": float, "recall": float, "f1": float},
            "hallucinated": {"precision": float, "recall": float, "f1": float}
        }
    """
    model.eval()
    example_preds: list[int] = []
    example_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating (Example Level)", leave=False):
            # Move inputs to device. Note that `batch["labels"]`
            # can stay on CPU if you wish to avoid unnecessary transfers.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits: torch.Tensor = (
                outputs.logits
            )  # Shape: [batch_size, seq_len, num_labels]
            predictions: torch.Tensor = torch.argmax(
                logits, dim=-1
            )  # Shape: [batch_size, seq_len]

            # Process each example in the batch separately.
            for i in range(batch["labels"].size(0)):
                sample_labels = batch["labels"][i]  # [seq_len]
                sample_preds = predictions[i].cpu()  # [seq_len]
                valid_mask = sample_labels != -100

                if valid_mask.sum().item() == 0:
                    # If no valid tokens, assign default class (supported).
                    true_example_label = 0
                    pred_example_label = 0
                else:
                    # Apply the valid mask and bring labels to CPU if needed.
                    sample_labels = sample_labels[valid_mask].cpu()
                    sample_preds = sample_preds[valid_mask]

                    # If any token in the sample is hallucinated (1), consider the whole sample hallucinated.
                    true_example_label = 1 if (sample_labels == 1).any().item() else 0
                    pred_example_label = 1 if (sample_preds == 1).any().item() else 0

                example_labels.append(true_example_label)
                example_preds.append(pred_example_label)

    precision, recall, f1, _ = precision_recall_fscore_support(
        example_labels, example_preds, labels=[0, 1], average=None, zero_division=0
    )

    results: dict[str, dict[str, float]] = {
        "supported": {  # Class 0
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        },
        "hallucinated": {  # Class 1
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
        },
    }

    if verbose:
        report = classification_report(
            example_labels,
            example_preds,
            target_names=["Supported", "Hallucinated"],
            digits=4,
            zero_division=0,
        )
        print("\nDetailed Example-Level Classification Report:")
        print(report)
        results["classification_report"] = report

    return results


def print_example_level_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """
    Print example-level evaluation metrics in a readable format.
    """
    print("\nExample-Level Evaluation Results:")
    print("\nHallucination Detection (Example Level) - Class 1:")
    print(f"  Precision: {metrics['hallucinated']['precision']:.4f}")
    print(f"  Recall: {metrics['hallucinated']['recall']:.4f}")
    print(f"  F1: {metrics['hallucinated']['f1']:.4f}")

    print("\nSupported Content (Example Level) - Class 0:")
    print(f"  Precision: {metrics['supported']['precision']:.4f}")
    print(f"  Recall: {metrics['supported']['recall']:.4f}")
    print(f"  F1: {metrics['supported']['f1']:.4f}")
