# LettuceDetect 🥬🔍

<p align="center">
  <img src="assets/lettuce_detective.png" alt="LettuceDetect Logo" width="400"/>
</p>

LettuceDetect is a hallucination detection tool designed to improve the reliability of AI-generated content. It helps identify parts of AI responses that are not supported by the given context.

## Features

- Token-level hallucination detection
- Support for transformer-based models
- Both token and span-level predictions
- Easy-to-use Python API
- Command-line interface for training and evaluation

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from lettucedetect.models.inference import HallucinationDetector

# Initialize the detector with the transformer-based approach.
detector = HallucinationDetector(method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")


context = """Briefly answer the following question:
What is the capital of France? What is the population of France?

Bear in mind that your response should be strictly based on the following three passages:
passage 1: France is a country in Europe. The capital of France is Paris. The population of France is 67 million.

output:
"""

answer = "The capital of France is Paris. The population of France is 69 million."

# Get span-level predictions indicating which parts of the answer are considered hallucinated.
predictions = detector.predict(context, answer, output_format="spans")
print("Predictions:", predictions)

# Predictions: [{'start': 31, 'end': 71, 'confidence': 0.9944414496421814, 'text': ' The population of France is 69 million.'}]
```

## Training a Model

```bash
python -m scripts.train \
    --data_path data/ragtruth/ragtruth_data.json \
    --model_name answerdotai/ModernBERT-base \
    --output_dir outputs/hallucination_detector \
    --batch_size 8 \
    --epochs 6
```

## Evaluation

```bash
python -m scripts.evaluate \
    --model_path outputs/hallucination_detector \
    --data_path data/ragtruth/ragtruth_data.json
```

## Model Output Format

The model can output predictions in two formats:

### Span Format
```python
[{
    'text': str,        # The hallucinated text
    'start': int,       # Start position in answer
    'end': int,         # End position in answer
    'confidence': float # Model's confidence (0-1)
}]
```

### Token Format
```python
[0, 0, 0, 1, 1, 0]  # 0: supported, 1: hallucinated
```

## License

MIT License - see LICENSE file for details.

## Citation

TODO: its coming soon until that:

```bibtex
@software{lettucedetect2024,
  title = {LettuceDetect: A Tool for Detecting Hallucinations in AI-Generated Content},
  author = {Kovacs, Adam},
```