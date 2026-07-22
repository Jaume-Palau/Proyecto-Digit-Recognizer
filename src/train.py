import pickle
import time
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.config import CHECKPOINTS_DIR, MODELS_DIR, PICKLE_DIR, TRAIN_CONFIG
from src.dataset import Dataset_Train
from src.models.model import Model
from src.helper_functions import set_seed


def train_model(model, config: dict,):

    set_seed(config["seed"])

    device = torch.device(config["device"])
    model = model.to(device)

    # =========================================================
    # INSTANCIACIÓN DEL DATASET
    # Aquí puedes cambiar Dataset_Train o sus argumentos.
    # =========================================================
    full_dataset = Dataset_Train()

    indices = list(range(len(full_dataset)))

    # =========================================================
    # SPLIT ESTRATIFICADO
    # Mantiene la proporción de cada dígito en train y validación.
    # =========================================================
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config["val_split"],
        random_state=config["seed"],
        shuffle=True,
        stratify=full_dataset.labels
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # =========================================================
    # DATALOADERS
    # Aquí puedes cambiar batch_size, num_workers, pin_memory...
    # =========================================================
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda"
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda"
    )

    # =========================================================
    # LOSS Y OPTIMIZADOR
    # Aquí puedes sustituir Adam por SGD, AdamW, etc.
    # =========================================================
    criterion = model.loss_fn

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )


    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINTS_DIR / config["name"] / "last_checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_model_path = MODELS_DIR / config["name"] / "best_model.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    history_path = PICKLE_DIR / config["name"] / "training_history.pkl"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    config["checkpoint_dir"] = str(checkpoint_path)
    config["model_dir"] = str(best_model_path)
    config["pickle_dir"] = str(history_path)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time": []
    }

    best_val_accuracy = 0.0
    total_start_time = time.perf_counter()

    epoch_bar = tqdm(
        range(1, config["n_epochs"] + 1),
        desc="Entrenamiento"
    )

    for epoch in epoch_bar:

        epoch_start_time = time.perf_counter()

        # =====================================================
        # ENTRENAMIENTO
        # =====================================================
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config['n_epochs']} - Train",
            leave=False
        )

        for images, labels in train_bar:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            train_loss_sum += loss.item() * batch_size
            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += batch_size

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        # =====================================================
        # VALIDACIÓN
        # =====================================================
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{config['n_epochs']} - Validation",
            leave=False
        )

        with torch.no_grad():

            for images, labels in val_bar:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                logits = model(images)
                loss = criterion(logits, labels)

                predictions = logits.argmax(dim=1)
                batch_size = labels.size(0)

                val_loss_sum += loss.item() * batch_size
                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += batch_size

        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total

        epoch_time = time.perf_counter() - epoch_start_time

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_time"].append(epoch_time)

        # =====================================================
        # GUARDAR MEJOR MODELO
        # =====================================================
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                best_model_path
            )

        # =====================================================
        # GUARDAR CHECKPOINT DE LA ÚLTIMA EPOCH
        # =====================================================
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "best_val_accuracy": best_val_accuracy,
                "config": config
            },
            checkpoint_path
        )

        epoch_bar.set_postfix(
            train_acc=f"{train_accuracy * 100:.2f}%",
            val_acc=f"{val_accuracy * 100:.2f}%",
            time=f"{epoch_time:.1f}s"
        )

    total_time = time.perf_counter() - total_start_time

    history["best_val_accuracy"] = best_val_accuracy
    history["total_time"] = total_time

    # =========================================================
    # GUARDAR HISTORIAL EN PICKLE
    # =========================================================
    with open(history_path, "wb") as file:
        pickle.dump(history, file)

    print(
        f"Mejor validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )
    print(f"Tiempo total: {total_time:.2f} segundos")

    return history


if __name__ == "__main__":
    
    history = train_model(model=Model(), config=TRAIN_CONFIG)