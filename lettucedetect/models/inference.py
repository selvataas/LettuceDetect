import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """
        Given a context and an answer, returns predictions.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        pass


class TransformerDetector(BaseDetector):
    def __init__(self, model_path: str, max_length: int = 4096, device=None, **kwargs):
        """
        Initialize the TransformerDetector.

        :param model_path: The path to the model.
        :param max_length: The maximum length of the input sequence.
        :param device: The device to run the model on.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, **kwargs
        )
        self.max_length = max_length
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """
        Performs inference and returns either token-level predictions or grouped spans.
        This version mimics the dataset's preprocessing: it tokenizes the prompt and answer,
        computes the answer start token (based on the prompt length), masks context tokens,
        and then processes only answer tokens.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: "tokens" or "spans".

        :return: If "tokens", a list of integer predictions for each token.
                 If "spans", a list of dictionaries, each with keys "label", "start", "end", "confidence", and "text".
        """
        # Tokenize context and answer with offsets.
        encoding = self.tokenizer(
            context,
            answer,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        # Save offset mapping for later use.
        offsets = encoding.pop("offset_mapping")[0]  # shape: (seq_length, 2)

        # Compute answer start token index:
        prompt_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        # Tokenization adds [CLS] at start and [SEP] after the prompt:
        answer_start_token = 1 + len(prompt_tokens) + 1  # [CLS] + prompt tokens + [SEP]

        # Create a label tensor: mark tokens before answer as -100 (ignored) and answer tokens as 0.
        labels = torch.full_like(encoding.input_ids[0], -100)
        labels[answer_start_token:] = 0

        # Move encoding to the device.
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        # Run model inference.
        with torch.no_grad():
            outputs = self.model(**encoding)
        logits = outputs.logits  # shape: (1, seq_length, num_classes)

        # Compute predictions and probabilities.
        token_preds = torch.argmax(logits, dim=-1)[0]  # shape: (seq_length,)
        probabilities = torch.softmax(logits, dim=-1)[
            0
        ]  # shape: (seq_length, num_classes)

        # Mask out predictions for context tokens.
        token_preds = torch.where(labels == -100, labels, token_preds)

        if output_format == "tokens":
            return token_preds.cpu().numpy().tolist()
        elif output_format == "spans":
            # Compute the answer's character offset (the first token of the answer).
            if answer_start_token < offsets.size(0):
                answer_char_offset = offsets[answer_start_token][0].item()
            else:
                answer_char_offset = 0

            spans: list[dict] = []
            current_span: dict | None = None

            # Iterate over tokens in the answer region.
            for i in range(answer_start_token, token_preds.size(0)):
                # Skip tokens marked as ignored.
                if labels[i].item() == -100:
                    continue

                token_start, token_end = offsets[i].tolist()
                # Skip special tokens with zero length.
                if token_start == token_end:
                    continue

                # Adjust offsets relative to the answer text.
                rel_start = token_start - answer_char_offset
                rel_end = token_end - answer_char_offset

                is_hallucination = (
                    token_preds[i].item() == 1
                )  # assuming class 1 indicates hallucination.
                confidence = probabilities[i, 1].item() if is_hallucination else 0.0

                if is_hallucination:
                    if current_span is None:
                        current_span = {
                            "start": rel_start,
                            "end": rel_end,
                            "confidence": confidence,
                        }
                    else:
                        # Extend the current span.
                        current_span["end"] = rel_end
                        current_span["confidence"] = max(
                            current_span["confidence"], confidence
                        )
                else:
                    # If we were building a hallucination span, finalize it.
                    if current_span is not None:
                        # Extract the hallucinated text from the answer.
                        span_text = answer[current_span["start"] : current_span["end"]]
                        current_span["text"] = span_text
                        spans.append(current_span)
                        current_span = None

            # Append any span still in progress.
            if current_span is not None:
                span_text = answer[current_span["start"] : current_span["end"]]
                current_span["text"] = span_text
                spans.append(current_span)

            return spans
        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")


class HallucinationDetector:
    def __init__(self, method: str = "transformer", **kwargs):
        """
        Facade for the hallucination detector.

        :param method: "transformer" for the model-based approach.
        :param kwargs: Additional keyword arguments passed to the underlying detector.
        """
        if method == "transformer":
            self.detector = TransformerDetector(**kwargs)
        else:
            raise ValueError("Unsupported method. Choose 'transformer'.")

    def predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        return self.detector.predict(context, answer, output_format)
