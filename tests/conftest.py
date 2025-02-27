"""Shared fixtures for pytest tests."""
import pytest
from unittest.mock import MagicMock
import torch


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 102, 103, 104, 105]
    
    # Mock tokenizer call to return encoding
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        "offset_mapping": torch.tensor([
            [0, 0],  # [CLS]
            [0, 4],  # "This"
            [5, 7],  # "is"
            [8, 9],  # "a"
            [10, 16],  # "prompt"
            [0, 0],  # [SEP]
            [0, 4],  # "This"
            [5, 12],  # "answer"
        ]),
    }
    
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]])
    model.return_value = mock_output
    return model 