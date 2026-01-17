import cv2
import json
import os
import gc
import uuid
import yaml
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Thêm thư viện này
from PIL import Image
from torchvision import transforms

# Import model và normalizer của Long
from src.models.efficient_transformer import EfficientNetTransformer
from src.preprocessor.normalizer import DataNormalizer

app = FastAPI(title="Human Action Recognition API - Optimized")

# ====================================================================
# CẤU HÌNH CORS 
# ====================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các nguồn (có thể thay bằng list domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"], # Cho phép tất cả các phương thức (GET, POST,...)
    allow_headers=["*"], # Cho phép tất cả các headers
)

# ====================================================================
# CONFIGURATION & MODEL LOADING
# ====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running on: {device} ---")

# Load labels logic (Giữ nguyên của bạn)
CLASS_MAP_PATH = "data/training_dataset/class_map.json"
try:
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_data = json.load(f)
        LABELS = class_data["idx_to_class"] 
        NUM_CLASSES = len(LABELS)
except Exception as e:
    print(f"Error loading labels: {e}")
    LABELS = [
        "brush_hair", "climb_stairs", "eat", 
        "fall_floor", "push", "walk"
    ]
    NUM_CLASSES = 6

models_pool = {}

def preload_models():
    path_ts_weight = "models/checkpoints/efficient_transformer/best_model_efficient_transformer.pth"
    path_ts_cfg = "configs/models/efficient_transformer.yml"
    
    if os.path.exists(path_ts_weight) and os.path.exists(path_ts_cfg):
        with open(path_ts_cfg, 'r') as f:
            cfg = yaml.safe_load(f)
        
        m_ts = EfficientNetTransformer(
            num_classes=NUM_CLASSES,
            d_model=cfg.get('d_model', 512),
            num_layers=cfg.get('num_layers', 2),
            nhead=cfg.get('nhead', 8),
            dropout=cfg.get('dropout', 0.5)
        )
        m_ts.load_state_dict(torch.load(path_ts_weight, map_location=device, weights_only=True))
        models_pool["cnn-transformer"] = m_ts.to(device).eval()
        print("Success: Loaded CNN-Transformer model.")

preload_models()

# Khởi tạo Normalizer
normalizer = DataNormalizer(output_size=224, yolo_path='./models/yolov8n.pt', batch_size=16)

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =======================================================================
# MAIN API LOGIC
# =======================================================================
@app.get("/")
async def root():
    return {"message": "API Human Action Recognition đang chạy!", "status": "Active"}

@app.post("/predict")
async def predict(video: UploadFile = File(...), model_type: str = Form("cnn-transformer")):
    if model_type not in models_pool:
        raise HTTPException(status_code=400, detail="Model type not supported.")

    unique_id = str(uuid.uuid4())
    input_path = f"temp_{unique_id}.mp4"
    
    # Save video tạm thời
    with open(input_path, "wb") as f:
        f.write(await video.read())

    try:
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            raise ValueError("Video file is empty or corrupted.")
            
        # Sampling 64 frames
        indices = np.linspace(0, total_frames - 1, 64, dtype=int)
        raw_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            raw_frames.append(frame)
        cap.release()

        # Tích hợp Normalizer
        processed_frames = []
        for i in range(0, len(raw_frames), normalizer.batch_size):
            batch = raw_frames[i : i + normalizer.batch_size]
            results = normalizer.model(batch, verbose=False, stream=True)
            
            for j, res in enumerate(results):
                person_crop = normalizer._get_person_crop(batch[j], res)
                final_frame = normalizer._pad_to_square(person_crop)
                final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                processed_frames.append(val_transform(Image.fromarray(final_frame)))

        # Padding nếu thiếu frame
        while len(processed_frames) < 64:
            processed_frames.append(processed_frames[-1] if processed_frames else torch.zeros(3, 224, 224))
        
        # Inference
        input_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = models_pool[model_type](input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, idx = torch.max(probs, 1)

        return {
            "label": LABELS[idx.item()] if idx.item() < len(LABELS) else "Unknown",
            "confidence": round(float(conf.item()), 4),
            "processed_frames": len(processed_frames)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if os.path.exists(input_path): os.remove(input_path)
        if 'input_tensor' in locals(): del input_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()