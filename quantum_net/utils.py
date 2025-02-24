import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import logging

logger = logging.getLogger(__name__)


def load_ag_news(file_path, max_length=128, vocab_size=30000):
    try:
        logger.info(f"Reading data from {file_path}")
        df = pd.read_csv(file_path, header=None, encoding="utf-8")
        df.columns = ["label", "title", "description"]
        df["text"] = df["title"] + " " + df["description"]
        df["label"] = df["label"] - 1

        logger.info("Building vocabulary...")
        vectorizer = CountVectorizer(
            max_features=vocab_size - 2,
            token_pattern=r"\b\w+\b",
        )
        vectorizer.fit(df["text"])
        vocab = {"<pad>": 0, "<unk>": 1}
        vocab.update(
            {
                word: idx + 2
                for idx, word in enumerate(vectorizer.get_feature_names_out())
            }
        )

        logger.info("Tokenizing and padding sequences...")
        X = df["text"].apply(lambda x: tokenize_and_pad(x, vocab, max_length)).tolist()
        y = df["label"].values
        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(
            f"Data loaded successfully. Train size: {len(X_train)}, Test size: {len(X_test)}"
        )
        return X_train, X_test, y_train, y_test, vocab

    except Exception as e:
        logger.error(f"Error loading AG News dataset: {str(e)}")
        raise e


def tokenize_and_pad(text, vocab, max_length):
    """Tokenize and pad/truncate a text sequence."""
    # Split text into words
    words = text.lower().split()

    # Convert words to indices
    tokens = [vocab.get(word, vocab["<unk>"]) for word in words[:max_length]]

    # Pad if necessary
    if len(tokens) < max_length:
        tokens.extend([vocab["<pad>"]] * (max_length - len(tokens)))

    return tokens


class TextDataset(Dataset):
    """Custom Dataset for text data."""

    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    """Create DataLoaders for training and testing."""
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Adjust based on your system
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Adjust based on your system
    )

    return train_loader, test_loader
