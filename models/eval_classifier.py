"""
To Evaluate the trained classifier model
"""
from binary_classifier import BinaryClassifier
from train_classifier import load_data, test
from pathlib import Path
import torch

    
def print_confusion_matrix(confusion_matrix: torch.Tensor) -> None:
    """
    Print a labeled confusion matrix for failure and success predictions.

    Args:
        confusion_matrix: Matrix with actual labels as rows and predicted
            labels as columns, using failure=0 and success=1.
    """
    class_names = ("failure", "success")
    print("Confusion matrix (rows: actual, columns: predicted):")
    print(f"{'':>12}{class_names[0]:>12}{class_names[1]:>12}")
    for class_index, class_name in enumerate(class_names):
        values = confusion_matrix[class_index]
        print(
            f"{class_name:>12}{int(values[0]):>12}{int(values[1]):>12}"
        )


if __name__ == "__main__":
    ####################################### HYPERPARAMETERS ########################################
    BATCH_SIZE = 10
    ################################################################################################

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    data_dir = Path(__file__).resolve().parents[1] / "data" / "classifier"
    model_dir = Path(__file__).resolve().parents[1] / "models" / "classifier" / "binary_classifier.pth"


    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load(model_dir, weights_only=True))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    _, test_dataloader = load_data(data_dir, BATCH_SIZE, 0)  # All test data
    
    loss, accuracy, confusion_matrix = test(test_dataloader, model, loss_fn, device)

    # TODO (@schromya): For evaluating generalization, this should really be tested on 
    # untrained data
    print(f"Evaluation Accuracy: {(100*accuracy):>0.1f}%, Evaluation loss: {loss:>8f}")

    print_confusion_matrix(confusion_matrix)
