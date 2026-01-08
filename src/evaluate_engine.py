import torch
import torch.nn as nn
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

from src.models.efficient_transformer import EfficientNetTransformer
from src.trainer.data_module import get_dataloaders

def load_model_full(config):
    class_map_path = Path(config["data_root"]) / "class_map.json"
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    num_classes = len(class_map["idx_to_class"])

    model = EfficientNetTransformer(
        num_classes=num_classes,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        nhead=config["nhead"],
        dropout=config.get("dropout", 0.3)
    )
    checkpoint_path = Path(config["checkpoints_dir"]) / "best_model.pth"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
        model.load_state_dict(state_dict)
        print(f">>> Đã load thành công checkpoint từ: {checkpoint_path}")
    else:
        print(">>> CẢNH BÁO: Không tìm thấy file checkpoint!")
    model.to(config["device"])
    model.eval()
    return model, class_map

def get_prediction_df(model, dataloader, class_map, device):
    idx_to_class = class_map["idx_to_class"]
    results = []

    print("--- Đang thực hiện dự đoán trên tập dữ liệu ---")
    with torch.no_grad():
        for videos, labels in tqdm(dataloader):
            videos = videos.to(device)
            outputs = model(videos)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(labels)):
                true_idx = labels[i].item()
                pred_idx = preds[i].item()
                results.append({
                    "True_Label": idx_to_class[true_idx],
                    "Pred_Label": idx_to_class[pred_idx],
                    "Confidence": round(probs[i][pred_idx].item(), 4),
                    "Correct": true_idx == pred_idx
                })

    return pd.DataFrame(results)

if __name__ == "__main__":
    from src.train_engine import CONFIG 

    _, _, test_loader, _ = get_dataloaders(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        num_frames=CONFIG["num_frames"]
    )

    model, class_map = load_model_full(CONFIG)
    df = get_prediction_df(model, test_loader, class_map, CONFIG["device"])

    print("\n" + "="*30)
    print("BÁO CÁO ĐỘ CHÍNH XÁC THEO LỚP")
    print("="*30)
    class_report = df.groupby("True_Label")["Correct"].mean().sort_values(ascending=False)
    print(class_report)
    
    df.to_csv("./models/checkpoints/detailed_eval_report.csv", index=False)
    print("\n>>> Đã lưu báo cáo chi tiết vào file: detailed_eval_report.csv")