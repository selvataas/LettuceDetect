import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthSample, RagTruthData
from pathlib import Path
import json


class RagTruthDataset(Dataset):
    """Dataset for RAG truth data."""

    def __init__(self, samples: list[RagTruthSample], tokenizer: AutoTokenizer, max_length: int = 8192):
        """Initialize the dataset.
        
        :param samples: List of RagTruthSample objects.
        :param tokenizer: Tokenizer to use for encoding the data.
        :param max_length: Maximum length of the input sequence.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get an item from the dataset.
        
        :param idx: Index of the item to get.
        :return: Dictionary with input IDs, attention mask, and labels.
        """
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample.prompt,
            sample.answer,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # Get the offset mapping as a list of (start, end) pairs
        offsets = encoding.pop("offset_mapping")[0]

        prompt_encoding = self.tokenizer.encode(sample.prompt, add_special_tokens=False)
        answer_start = 1 + len(prompt_encoding) + 1

        # Initialize token labels as -100 (ignore index)
        labels = [-100] * encoding.input_ids.shape[1]

        # For tokens in the answer region, compute the label based on overlap with annotated spans
        # We adjust the character offsets to be relative to the answer_text.
        answer_char_offset = offsets[answer_start][0] if answer_start < len(offsets) else None

        for i in range(answer_start, encoding.input_ids.shape[1]):
            token_start, token_end = offsets[i]
            # Adjust token offset relative to answer_text
            token_abs_start = token_start - answer_char_offset
            token_abs_end = token_end - answer_char_offset

            # Default label is 0 (supported)
            token_label = 0
            # if token overlaps any annotated hallucination span, mark it as unsupported, hallucination (1)
            for ann in sample.labels:
                if token_abs_end > ann['start'] and token_abs_start < ann['end']:                
                    token_label = 1
                    break

            labels[i] = token_label

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "labels": labels,
        }
        

if __name__ == "__main__":
    input_dir = Path("data/ragtruth/ragtruth_data.json")
    rag_truth_data = RagTruthData.from_json(json.loads(input_dir.read_text()))

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    dataset = RagTruthDataset(rag_truth_data.samples, tokenizer)

    # Get a sample and decode it to verify alignment
    sample_idx = 2
    sample_data = dataset[sample_idx]
    original_sample = rag_truth_data.samples[sample_idx]
    
    # Decode tokens and print with their labels
    tokens = tokenizer.convert_ids_to_tokens(sample_data["input_ids"])
    labels = sample_data["labels"]
    
    print("Original sample:")
    print(f"Prompt: {original_sample.prompt}")
    print(f"Answer: {original_sample.answer}")
    print("\nHallucinated spans:")
    print(original_sample.labels)
    # Hallucinated text:
    for label in original_sample.labels:
        print(original_sample.answer[label['start']:label['end']])
    
    print("\nToken-level annotations:")
    for token, label in zip(tokens, labels):
        if label != -100:  # Only print tokens we're actually labeling
            print(f"{token}: {'hallucinated' if label == 1 else 'supported'}")

    # Print token sequence breakdown
    prompt_tokens = tokenizer.encode(original_sample.prompt, add_special_tokens=False)
    answer_start = 1 + len(prompt_tokens) + 1
    
    print("\nToken sequence:")
    print("Prompt tokens:", tokens[:answer_start])
    print("Answer tokens:", tokens[answer_start:])
