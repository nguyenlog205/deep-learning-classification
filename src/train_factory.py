import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
import yaml
import sys

from src.trainer.data_module import get_dataloaders
from src.trainer.training import train, evaluate
from src.models.efficient_transformer import EfficientNetTransformer
from src.models.efficient_mamba import EfficientNetMamba

# ===========================================================
# CONFIGURATION
# ===========================================================

# --- From configs ---
def get_configs(model_name, config_dir='./configs/models/'):
    config_dir = Path(config_dir)
    config_path = config_dir / f"{model_name}.yaml"

    if not config_path.exists():
        config_path = config_dir / f"{model_name}.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file config cho model '{model_name}' tại: {config_dir}")

    print(f"📂 Loading config from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config

# --- From parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Training Factory for Video Classification")
    parser.add_argument("--model", type=str, default="efficient_transformer", 
                        choices=["efficient_transformer", "efficient_mamba"], 
                        help="Chọn model để train")
    return parser.parse_args()

# ===============================================================
# Model Preparation 
# ===============================================================

def setup_model(model_name, num_classes, config):
    print(f"\n⚙️  SETTING UP MODEL: [ {model_name.upper()} ]")
    device = config['device']

    # 1. Init Architecture
    if model_name == "efficient_transformer":
        model = EfficientNetTransformer(
            num_classes=num_classes, d_model=config["d_model"],
            nhead=config["nhead"], num_layers=config["num_layers"], dropout=config["dropout"]
        )
    elif model_name == "efficient_mamba":
        model = EfficientNetMamba(
            num_classes=num_classes, d_model=config["d_model"],
            num_layers=config["num_layers"], d_state=config["d_state"],
            d_conv=config["d_conv"], expand=config["expand"], dropout=config["dropout"]
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)

    # 2. Auto Resume Checkpoint
    ckpt_path = Path(config["checkpoints_dir"]) / f"best_model_{model_name}.pth"
    if ckpt_path.exists():
        print(f"   🔄 Resuming from: {ckpt_path}")
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        except Exception as e:
            print(f"   ⚠️ Load failed ({e}). Starting from scratch.")
    else:
        print(f"   🆕 No checkpoint found. Starting from scratch.")

    # 3. Freeze Backbone (Logic tích hợp luôn tại đây)
    if hasattr(model, 'backbone'):
        print("   🔒 Freezing backbone (keeping last 2 blocks open)...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.features[-2:].parameters():
            param.requires_grad = True
    
    # Report Params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model Ready. Trainable params: {trainable:,}")
    
    return model
    
def main():
    args = get_args()
    try:
        CONFIG = get_configs(model_name=args.model) 
    except Exception as e:
        print(f"❌ Config Error: {e}")
        sys.exit(1)
    CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n✅ READY TO TRAIN: {args.model.upper()}")
    print(f"   Device:     {CONFIG['device'].upper()}")
    print(f"   Batch Size: {CONFIG['batch_size']}")
    print(f"   Epochs:     {CONFIG['epochs']}")
    print("-" * 40)
    
    # === PREPARE DATA ===
    print("\n[1/5] Loading Datasets...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        num_frames=CONFIG["num_frames"],
        num_workers=CONFIG["num_workers"]
    )
    print(f"   >>> Found {num_classes} classes.")
    
    # === INIT MODEL ===
    print("\n[2/5] Initializing Model Architecture...")
    model = setup_model(args.model, num_classes, CONFIG)
    
    # === SETUP OPTIMIZER & LOSS ===
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )
    
    # === START TRAINING ===
    ckpt_name = f"best_model_{args.model}.pth"
    Path(CONFIG["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    
    print(f"\n[3/4] Starting Training Loop for {args.model}...")
    
    train(
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
        use_amp=CONFIG["use_amp"],
        save_name=ckpt_name
    )

    # === FINAL EVALUATION ===
    print(f"\n[4/4] Evaluating Best Model on Test Set...")
    best_ckpt_path = Path(CONFIG["checkpoints_dir"]) / ckpt_name
    
    if best_ckpt_path.exists():
        # Load lại trọng số tốt nhất vừa train xong để test
        print(f"   >>> Loading best weights from {ckpt_name}...")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=CONFIG["device"]))
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, torch.device(CONFIG["device"]))
        print(f"   🏆 TEST RESULT [{args.model.upper()}]: Loss = {test_loss:.4f} | Accuracy = {test_acc:.4f}")
    else:
        print("   ⚠️ Warning: No best model found to evaluate.")

if __name__ == "__main__":
    main()
