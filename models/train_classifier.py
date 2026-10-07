"""
Reference:
https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/quickstart_tutorial.py
"""
from binary_classifier import BinaryClassifier

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_data(data_dir, batch_size=10, train_ratio=0.5):
    """
    Load labeled images and create a reproducible 80/20 train/test split.
    Args:
        data_dir: Directory containing failure/ and success/ image folders.
        batch_size: Number of images in each batch.
        train_ratio: Ratio of data for the training split, from 0 to 1.
    Returns:
        Training and test DataLoaders with images resized to 224 x 224.
    """

    # TODO (schromya): Crop image to 224x224 to begin with?
    # NOTTODO(abhismirai42): Croping image is distroying the information, resize is better right?
    #                        Also we need to make sure that same croping/resizing applied while eval
    # (schromya): I meant cropping smaller in data processing (data/prep_classifier_data)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=preprocess)
    if dataset.class_to_idx != {"failure": 0, "success": 1}:
        raise ValueError("Expected only failure/ and success/ folders in the dataset.")
    if len(dataset) < 2:
        raise ValueError("Need at least two images for a train/test split.")
    if not 0 <= train_ratio <= 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    training_data, test_data = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=len(training_data) > 0
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=len(test_data) > 0
    )
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer, device) -> float:
    """
    Train the model for one complete pass through the dataset.

    Args:
        dataloader: Batches of input images and binary labels.
        model: Model being optimized.
        loss_fn: Loss function used to compare predictions with labels.
        optimizer: Optimizer used to update the model's trainable parameters.
        device: Device where input tensors and the model are stored.
    Returns:
        Training loss of last batch.
    """
    size = len(dataloader.dataset)
    model.train()
    model.backbone.eval()  # Keep the frozen backbone's BatchNorm statistics fixed.
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device).float().reshape(-1, 1)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, _ = loss.item(), min((batch + 1) * dataloader.batch_size, size)
    return loss

def test(dataloader, model, loss_fn, device) -> tuple[float, float, torch.Tensor]:
    """
    Evaluate the model on one complete pass through the test dataset.

    Args:
        dataloader: Batches of input images and binary labels.
        model: Model being evaluated.
        loss_fn: Loss function used to compare predictions with labels.
        device: Device where input tensors and the model are stored.

    Returns:
    A tuple containing:
        - average test loss as a float.
        - accuracy as a float from 0 to 1.
        - a 2x2 confusion matrix with actual labels as rows and predicted labels as columns, 
          using failure=0 and success=1.
    """
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    confusion_matrix = torch.zeros((2, 2), dtype=torch.int64)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float().reshape(-1, 1)
            labels = y.to(device).long().reshape(-1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * X.size(0)
            predicted_labels = (pred.reshape(-1) >= 0)
            correct += (predicted_labels == labels.bool()).sum().item()
            for actual, predicted in zip(labels.cpu(), predicted_labels.cpu()):
                confusion_matrix[int(actual.item()), int(predicted.item())] += 1

    test_loss /= size
    correct /= size

    return test_loss, correct, confusion_matrix


if __name__ == "__main__":

    ####################################### HYPERPARAMETERS ########################################
    BATCH_SIZE = 10
    EPOCHS = 100
    TRAIN_RATIO = 0.5  # Even split for smaller datasets (100) for better representation of data
    ################################################################################################
    
    batch_size = 10
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    data_dir = Path(__file__).resolve().parents[1] / "data" / "classifier"
    model_dir = Path(__file__).resolve().parents[1] / "models" / "classifier" / "binary_classifier.pth"
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader = load_data(data_dir, BATCH_SIZE, TRAIN_RATIO)

    model = BinaryClassifier().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-4)


    for i in range(EPOCHS):
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        loss, accuracy, _ = test(test_dataloader, model, loss_fn, device)

        if i % 20 == 0 or i == EPOCHS-1:
            print(f"[Epoch {i+1}/{EPOCHS}] Train loss: {train_loss:8f}, Test Accuracy: {(100*accuracy):.1f}%, Test Avg loss: {loss:8f}")

    torch.save(model.state_dict(), model_dir)
    print(f"\nSaved weights to {model_dir}")