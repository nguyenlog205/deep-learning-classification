import cv2
import numpy as np
import os
import gc
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from src.utils import load_dataset 

class DataNormalizer:
    def __init__(
        self,
        output_size: int = 224,     
        yolo_path: str = './models/yolov8n.pt',
        context_padding: float = 0.15,
        output_root: str = './data/processed_videos/',
        batch_size: int = 16  # <--- THÊM: Xử lý từng cụm 16 frame để không tràn RAM
    ):
        self.output_size = output_size
        self.context_padding = context_padding
        self.output_root = Path(output_root)
        self.batch_size = batch_size
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading YOLO from {yolo_path}...")
        self.model = YOLO(yolo_path)

    def _pad_to_square(self, image):
        """Resize và thêm padding đen để ảnh vuông mà không méo"""
        h, w = image.shape[:2]
        target = self.output_size
        
        # Tính scale
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Canvas đen
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        
        # Căn giữa
        x_off = (target - new_w) // 2
        y_off = (target - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        return canvas

    def _get_person_crop(self, frame, result):
        """Cắt người từ kết quả YOLO của 1 frame"""
        h_img, w_img = frame.shape[:2]
        roi = frame # Fallback: lấy cả ảnh nếu không thấy người
        
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == 0: # Class 0 là Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Padding thêm bối cảnh
                w_box, h_box = x2 - x1, y2 - y1
                pad_w = int(w_box * self.context_padding)
                pad_h = int(h_box * self.context_padding)
                
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w_img, x2 + pad_w)
                y2 = min(h_img, y2 + pad_h)
                
                roi = frame[y1:y2, x1:x2]
                return roi # Lấy người đầu tiên thấy
        return roi

    def process_one_video(self, row):
        video_path = row['video_path']
        save_dir = self.output_root / row['label']
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = str(save_dir / f"{row['id']}.mp4")
        if os.path.exists(save_path): return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0

        # Init Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (self.output_size, self.output_size))

        # --- BATCH PROCESSING LOOP ---
        while True:
            batch_frames = []
            # 1. Đọc n frames (Batching)
            for _ in range(self.batch_size):
                ret, frame = cap.read()
                if not ret: break
                batch_frames.append(frame)
            
            if not batch_frames: break

            # 2. Detect cả batch (Nhẹ hơn detect cả video)
            try:
                results = self.model(batch_frames, verbose=False, stream=True)
                
                # 3. Xử lý từng frame trong batch
                for i, r in enumerate(results):
                    frame = batch_frames[i]
                    person_crop = self._get_person_crop(frame, r)
                    
                    if person_crop.size == 0:
                        person_crop = frame
                        
                    final_frame = self._pad_to_square(person_crop)
                    out.write(final_frame)
            
            except Exception as e:
                print(f"Error in batch: {e}")
                # Nếu lỗi VRAM, thử dọn dẹp
                torch.cuda.empty_cache()
                continue
            
            # Dọn dẹp RAM sau mỗi batch
            del batch_frames
            del results
            
        cap.release()
        out.release()
        
        # Dọn dẹp mạnh tay sau mỗi video để tránh leak memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, df):
        print(f"Start processing {len(df)} videos with Batch Size = {self.batch_size}...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                self.process_one_video(row)
            except Exception as e:
                print(f"Error processing {row['id']}: {e}")

def main():
    # Load dataset
    df = load_dataset('./data/origin/')
    
    # Giảm batch_size xuống 16 hoặc 8 nếu vẫn bị lỗi RAM
    normalizer = DataNormalizer(
        output_size=224, 
        context_padding=0.15,
        output_root='./data/processed/',
        batch_size=64 
    )
    normalizer.run(df)

if __name__ == "__main__":
    main()