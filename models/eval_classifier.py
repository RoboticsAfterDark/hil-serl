"""
To Evaluate the trained classifier model
"""
from binary_classifier import BinaryClassifier
from train_classifier import load_data, test
from pathlib import Path
import torch

    
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
    
    loss, accuracy = test(test_dataloader, model, loss_fn, device)

    # TODO (@schromya): For evaluating generalization, this should really be tested on 
    # untrained data
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
