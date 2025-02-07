import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthData
from pathlib import Path
import json
from transformers import DataCollatorForTokenClassification
from tqdm.auto import tqdm
import time
from datetime import timedelta
from lettucedetect.model.evaluate import evaluate_model, print_metrics
from lettucedetect.model.dataset import RagTruthDataset

if __name__ == "__main__":
    input_dir = Path("data/ragtruth/ragtruth_data.json")
    rag_truth_data = RagTruthData.from_json(json.loads(input_dir.read_text()))

    train = [sample for sample in rag_truth_data.samples if sample.split == "train"]
    test = [sample for sample in rag_truth_data.samples if sample.split == "test"]

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )

    train_dataset = RagTruthDataset(train, tokenizer)
    test_dataset = RagTruthDataset(test, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, collate_fn=data_collator
    )

    model = AutoModelForTokenClassification.from_pretrained(
        "answerdotai/ModernBERT-base", num_labels=2
    )
    optimizer = AdamW(model.parameters(), lr=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_f1 = 0
    start_time = time.time()

    print(f"\nStarting training on {device}")
    print(
        f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}\n"
    )

    for epoch in range(6):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/6")

        # Training
        model.train()
        total_loss = 0
        num_batches = 0

        # Create progress bar for training
        progress_bar = tqdm(train_loader, desc="Training", leave=True)

        for batch in progress_bar:
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / num_batches:.4f}",
                }
            )

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start

        # Evaluation
        print("\nEvaluating...")
        metrics = evaluate_model(model, test_loader, device)
        print_metrics(metrics)

        # Save best model based on hallucination F1
        if metrics["hallucinated"]["f1"] > best_f1:
            best_f1 = metrics["hallucinated"]["f1"]
            model.save_pretrained("best_model")
            tokenizer.save_pretrained("best_model")
            print(f"\n🎯 New best F1: {best_f1:.4f}, model saved!")

        print("-" * 50)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
    print(f"Best F1 score: {best_f1:.4f}")
