import torch
import torch.nn as nn
import pandas as pd
import json
import argparse
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# --- IMPORTS TỪ PROJECT CỦA BÁC ---
from src.trainer.data_module import get_dataloaders
from src.models.efficient_transformer import EfficientNetTransformer
from src.models.efficient_mamba import EfficientNetMamba

# =========================================================================
# 1. CÁC HÀM TIỆN ÍCH (CONFIG & ARGS)
# =========================================================================

def get_configs(model_name, config_dir='./configs/models/'):
    config_dir = Path(config_dir)
    config_path = config_dir / f"{model_name}.yaml"
    
    if not config_path.exists():
        # Fallback
        config_path = config_dir / f"{model_name}.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config not found: {config_path}")

    print(f"📂 Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation Factory")
    parser.add_argument("--model", type=str, default="efficient_mamba", 
                        choices=["efficient_transformer", "efficient_mamba"],
                        help="Chọn model cần đánh giá")
    parser.add_argument("--split", type=str, default="test", 
                        choices=["train", "val", "test"],
                        help="Chọn tập dữ liệu để đánh giá")
    return parser.parse_args()

# =========================================================================
# 2. MODEL LOADER (FACTORY PATTERN)
# =========================================================================

def load_model_for_eval(model_name, config, device):
    """
    Khởi tạo kiến trúc và load trọng số best_model_{model_name}.pth
    """
    # 1. Load Class Map để biết số lượng lớp
    class_map_path = Path(config["data_root"]) / "class_map.json"
    if not class_map_path.exists():
        raise FileNotFoundError(f"❌ Thiếu file class_map.json tại {class_map_path}")
        
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    num_classes = len(class_map["idx_to_class"])
    
    print(f"🏭 Initializing [ {model_name.upper()} ] for {num_classes} classes...")

    # 2. Khởi tạo kiến trúc (Phải khớp với config lúc train)
    if model_name == "efficient_transformer":
        model = EfficientNetTransformer(
            num_classes=num_classes,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
    elif model_name == "efficient_mamba":
        model = EfficientNetMamba(
            num_classes=num_classes,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            d_state=config["d_state"],
            d_conv=config["d_conv"],
            expand=config["expand"],
            dropout=config["dropout"]
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 3. Load Checkpoint
    # Đường dẫn file checkpoint chuẩn theo train_factory
    ckpt_name = f"best_model_{model_name}.pth"
    checkpoint_path = Path(config["checkpoints_dir"]) / ckpt_name

    if checkpoint_path.exists():
        print(f"🔄 Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("✅ Weights loaded successfully!")
    else:
        print(f"❌ CRITICAL: Không tìm thấy file checkpoint tại {checkpoint_path}")
        sys.exit(1)

    model.to(device)
    model.eval() # Quan trọng: Chuyển sang chế độ Eval (tắt Dropout)
    
    return model, class_map

# =========================================================================
# 3. EVALUATION ENGINE
# =========================================================================
def evaluate_engine(model, dataloader, class_map, device, output_dir):
    idx_to_class = class_map["idx_to_class"]
    y_true = []
    y_pred = []
    results = []
    
    # --- FIX LOGIC LẤY LABEL NAME ---
    # Kiểm tra xem idx_to_class là List hay Dict để lấy tên class cho đúng
    def get_label_name(idx, mapper):
        if isinstance(mapper, list):
            return mapper[idx] # Nếu là List: dùng index số nguyên (0, 1)
        else:
            return mapper.get(str(idx), str(idx)) # Nếu là Dict: dùng key string ("0", "1")

    print("--- 🚀 Running Inference ---")
    with torch.no_grad():
        for videos, labels in tqdm(dataloader):
            videos = videos.to(device)
            
            # Forward pass
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Collect data for metrics
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Collect detailed results
            for i in range(len(labels)):
                t = labels[i].item() # Integer
                p = preds[i].item()  # Integer
                
                results.append({
                    "True_ID": t,
                    "Pred_ID": p,
                    "True_Label": get_label_name(t, idx_to_class), # <--- Đã sửa
                    "Pred_Label": get_label_name(p, idx_to_class), # <--- Đã sửa
                    "Confidence": round(probs[i][p].item(), 4),
                    "Correct": t == p
                })

    # --- REPORTING ---
    df = pd.DataFrame(results)
    
    # 1. Save Raw Predictions
    csv_path = output_dir / "predictions_full.csv"
    df.to_csv(csv_path, index=False)
    
    # 2. Save Misclassified Only
    df_wrong = df[df["Correct"] == False]
    df_wrong.to_csv(output_dir / "predictions_wrong.csv", index=False)

    print("\n" + "="*50)
    print("📊 CLASSIFICATION REPORT")
    print("="*50)
    
    # --- FIX LOGIC TARGET NAMES ---
    if isinstance(idx_to_class, list):
        target_names = idx_to_class
    else:
        # Nếu là dict, sort theo key integer để đảm bảo đúng thứ tự 0, 1, 2...
        sorted_keys = sorted(idx_to_class.keys(), key=lambda x: int(x))
        target_names = [idx_to_class[k] for k in sorted_keys]
    
    # Generate Report
    # labels=... để đảm bảo report hiện đủ các class ngay cả khi tập test thiếu 1 vài class
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    try:
        report = classification_report(
            y_true, 
            y_pred, 
            labels=unique_labels,
            target_names=[target_names[i] for i in unique_labels], 
            digits=4
        )
        print(report)
        
        with open(output_dir / "classification_report.txt", "w") as f:
            f.write(report)
            
    except Exception as e:
        print(f"⚠️ Could not generate full report text: {e}")
        print("   (This usually happens if classes in Test set mismatch Class Map)")
        
    print(f"\n📂 Results saved to: {output_dir}")
    print(f"   - Full predictions: predictions_full.csv")
    print(f"   - Errors only:      predictions_wrong.csv")
    print(f"   - Metrics report:   classification_report.txt")

# =========================================================================
# MAIN
# =========================================================================

def main():
    args = get_args()
    
    # 1. Load Config
    try:
        CONFIG = get_configs(model_name=args.model)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")

    # 2. Load Data (Lấy đúng split cần test)
    print("\n[1/3] Loading Data...")
    # Gọi hàm get_dataloaders nhưng chỉ quan tâm loader cần dùng
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        num_frames=CONFIG["num_frames"],
        num_workers=CONFIG["num_workers"]
    )
    
    if args.split == 'test':
        target_loader = test_loader
    elif args.split == 'val':
        target_loader = val_loader
    else:
        target_loader = train_loader

    print(f"   >>> Evaluating on set: {args.split.upper()} | Samples: {len(target_loader.dataset)}")

    # 3. Load Model
    print("\n[2/3] Loading Model...")
    model, class_map = load_model_for_eval(args.model, CONFIG, device)

    # 4. Run Evaluation
    print("\n[3/3] Start Evaluation...")
    # Tạo thư mục output riêng cho từng model để không bị đè file
    output_dir = Path(CONFIG["checkpoints_dir"]) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_engine(model, target_loader, class_map, device, output_dir)

if __name__ == "__main__":
    main()