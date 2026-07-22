
import torch
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import Dataset_Test, Dataset_Train
from src.config import SUBMISSIONS_DIR, TRAIN_CONFIG
from src.models.model import Model






def load_trained_model(checkpoint_path, model):
    ## Load the model state from output file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_checkpoint = torch.load(
        checkpoint_path,  
    map_location=torch.device(device)
    )
    ## Load the model state
    model.load_state_dict(loaded_checkpoint)
    model.to(device)
    model.eval()

    return model


def create_submisssion(file_id,predictions,output_path=SUBMISSIONS_DIR):
    ## Save the data in CSV for submission
    df = pd.DataFrame(
        {"id":file_id, 
        "label":predictions},
        )
    df.to_csv(str(output_path), index=False)


def predict_test_set(model, config: dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_model_path = Path(
        config["model_dir"] / "best_model.pth"
    )

    model = load_trained_model(
        checkpoint_path=best_model_path,
        model=model
    )

    test_dataset = Dataset_Test(
        csv_path=config["test_csv"]
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device == "cuda"
    )

    predictions = []

    with torch.no_grad():

        for images in tqdm(
            test_loader,
            desc="Generando predicciones"
        ):
            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            predicted_classes = logits.argmax(dim=1)

            predictions.extend(
                predicted_classes.cpu().tolist()
            )

    return predictions


if __name__ == "__main__":

    predictions = predict_test_set(model = Model(), config = TRAIN_CONFIG)

    print("Predicciones generadas:", predictions[:50])  # Muestra las primeras 50 predicciones
