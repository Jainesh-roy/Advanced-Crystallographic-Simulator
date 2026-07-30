"""
trainer.py

Train the XRD CNN classifier.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .cnn_model import XRDClassifier
from .config import *


# ==========================================================
# Training Configuration
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 25


# ==========================================================
# Accuracy
# ==========================================================

def accuracy(outputs, labels):

    predictions = outputs.argmax(dim=1)

    return (predictions == labels).float().mean().item()


# ==========================================================
# Training Function
# ==========================================================

def train():

    # ------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------

    train_data = torch.load(
        DATASET_DIR / "train.pt",
        weights_only=False
    )

    validation_data = torch.load(
        DATASET_DIR / "validation.pt",
        weights_only=False
    )

    train_dataset = TensorDataset(
        train_data["X"].float().unsqueeze(1),
        train_data["y"].long()
    )

    validation_dataset = TensorDataset(
        validation_data["X"].float().unsqueeze(1),
        validation_data["y"].long()
    )

    materials = train_data["materials"]

    num_classes = len(materials)

    # ------------------------------------------------------
    # Data Loaders
    # ------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = XRDClassifier(
        num_classes=num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_accuracy = 0.0

    print("\n==============================")
    print("Training Started")
    print("==============================\n")

    # ------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------

    for epoch in range(EPOCHS):

        # --------------------------
        # Training
        # --------------------------

        model.train()

        train_loss = 0.0
        train_accuracy = 0.0

        for X, y in train_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            train_accuracy += accuracy(outputs, y)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        # --------------------------
        # Validation
        # --------------------------

        model.eval()

        validation_loss = 0.0
        validation_accuracy = 0.0

        with torch.no_grad():

            for X, y in validation_loader:

                X = X.to(DEVICE)
                y = y.to(DEVICE)

                outputs = model(X)

                loss = criterion(outputs, y)

                validation_loss += loss.item()

                validation_accuracy += accuracy(outputs, y)

        validation_loss /= len(validation_loader)
        validation_accuracy /= len(validation_loader)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        # --------------------------
        # Save Best Model
        # --------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "materials": materials,
                },
                MODEL_DIR / "xrd_classifier.pth",
            )

    print("\n==============================")
    print("Training Finished")
    print("==============================")
    print(f"Best Validation Accuracy : {best_accuracy:.4f}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train()