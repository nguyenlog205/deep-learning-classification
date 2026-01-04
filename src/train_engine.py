import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.trainer.data_module import get_dataloaders
from src.model import EfficientNetTransformer
from src.trainer.training import train, evaluate

# --- CONFIGURATION ---
CONFIG = {
    "data_root": "./data/training_dataset",
    "num_frames": 64,
    "batch_size": 2,
    "num_workers": 4,

    # Model params
    "d_model": 1280,
    "num_layers": 2,
    "nhead": 8,
    
    # Training params
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "patience": 10,
    "accumulation_steps": 16,
    "use_amp": True,
    
    # Paths
    "checkpoints_dir": "./checkpoints",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def main():
    print(f"--- RUNNING TRAINING ON {CONFIG['device'].upper()} ---")
    
    # Prepare Data
    print("\n[1/5] Loading Datasets...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        num_frames=CONFIG["num_frames"],
        num_workers=CONFIG["num_workers"]
    )
    print(f"   >>> Found {num_classes} classes.")

    # Prepare Model
    print("\n[2/5] Initializing EfficientNet-Transformer...")
    model = EfficientNetTransformer(
        num_classes=num_classes,
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        nhead=CONFIG["nhead"]
    )

    # Setup Loss & Optimizer
    print("\n[3/5] Setting up Optimizer & Loss...")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training
    print("\n[4/5] Starting Training Loop...")
    history = train(
        model=model,
        train_loader=train_loader,
        eval_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=CONFIG["epochs"],
        device=CONFIG["device"],
        patience=CONFIG["patience"],
        accumulation_steps=CONFIG["accumulation_steps"],
        checkpoints_dir=CONFIG["checkpoints_dir"],
        use_amp=CONFIG["use_amp"]
    )

    # Final Test Evaluation
    print("\n[5/5] Evaluating on Test Set (Best Model)...")
    best_model_path = Path(CONFIG["checkpoints_dir"]) / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print("   >>> Loaded best model weights.")
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, torch.device(CONFIG["device"]))
        print(f"   >>> TEST RESULT: Loss = {test_loss:.4f} | Accuracy = {test_acc:.4f}")
    else:
        print("   >>> Warning: No best model found to evaluate.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()
    # python -m src.train_engine