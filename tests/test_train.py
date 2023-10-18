"""Tests the training of the model."""


import train


def test_train() -> None:
    original_epochs = train.configuration.epochs
    train.configuration.epochs = 10
    try:
        result = train.train()
        assert result[-1]["aurocMean"] > 0.5
    finally:
        train.configuration.epochs = original_epochs
