"""
To Evaluate the trained classifier model
"""
from binary_classifier import BinaryClassifier

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_data(data_dir, batch_size=10):
    """
    Load labeled images and create a reproducible 80/20 train/test split.
    Args:
        data_dir: Directory containing failure/ and success/ image folders.
        batch_size: Number of images in each batch.
    Returns:
        Training and test DataLoaders with images resized to 224 x 224.
    """

    # TODO (schromya): Crop image to 224x224 to begin with?
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=preprocess)
    if dataset.class_to_idx != {"failure": 0, "success": 1}:
        raise ValueError("Expected only failure/ and success/ folders in the dataset.")
    if len(dataset) < 2:
        raise ValueError("Need at least two images for a train/test split.")

    test_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    # return train_dataloader, test_dataloader
    return test_dataloader

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float().reshape(-1, 1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * X.size(0)
            predicted_labels = pred >= 0
            correct += (predicted_labels == y.bool()).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    batch_size = 10
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    data_dir = Path(__file__).resolve().parents[1] / "data" / "classifier"
    model_dir = Path(__file__).resolve().parents[1] / "models" / "classifier" / "binary_classifier.pth"


    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load(model_dir, weights_only=True))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    test_dataloader = load_data(data_dir, batch_size)
    
    test(test_dataloader, model, loss_fn, device)
