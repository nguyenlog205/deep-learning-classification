import os
import shutil
import cv2
import numpy as np
import random
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- SETUP LOGGING ---
def setup_logger(log_file='./logging/augment.log'):
    """Cấu hình logging: Vừa in ra màn hình, vừa lưu vào file"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# --- AUGMENTOR ---
class VideoAugmentor:
    def __init__(self):
        pass

    def apply_gaussian_noise(self, frames, mean=0, sigma=25):
        aug_frames = []
        for f in frames:
            noisy = f.astype(np.float32)
            gauss = np.random.normal(mean, sigma, f.shape).astype(np.float32)
            noisy = noisy + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            aug_frames.append(noisy)
        return aug_frames

    def apply_cutout(self, frames, n_holes=3, length=40):
        aug_frames = []
        h, w = frames[0].shape[:2]
        for f in frames:
            mask = f.copy()
            for _ in range(n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                mask[y1:y2, x1:x2] = 0
            aug_frames.append(mask)
        return aug_frames

    def apply_flip(self, frames):
        return [cv2.flip(f, 1) for f in frames]

    def apply_speed(self, frames, factor=None):
        if not factor: factor = random.choice([0.75, 1.25])
        indices = np.linspace(0, len(frames) - 1, int(len(frames) / factor)).astype(int)
        return [frames[i] for i in indices]

    def apply_brightness(self, frames, factor=None):
        if not factor: factor = random.uniform(0.7, 1.3)
        aug_frames = []
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
            hsv = hsv.astype(np.uint8)
            aug_frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        return aug_frames

# --- (MANAGER) ---
class PipelineManager:
    def __init__(self, source_dir='./data/processed_videos/', output_dir='./data/dataset/'):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.augmentor = VideoAugmentor()
        logger.info(f"Initialized Pipeline: Source={source_dir}, Output={output_dir}")

    def _save_video(self, frames, save_path, fps=30.0):
        if not frames: return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()

    def _read_video(self, path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        return frames, fps

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        logger.info("========== STEP 1: SPLITTING DATASET ==========")
        
        all_videos = []
        labels = []
        
        if self.output_dir.exists():
            logger.warning(f"Removing existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        if not self.source_dir.exists():
            logger.error(f"Source directory not found: {self.source_dir}")
            return

        for label_dir in self.source_dir.iterdir():
            if not label_dir.is_dir(): continue
            for vid in label_dir.glob("*.mp4"):
                all_videos.append(vid)
                labels.append(label_dir.name)

        logger.info(f"Found total {len(all_videos)} videos across {len(set(labels))} classes.")

        X_train, X_temp, y_train, y_temp = train_test_split(
            all_videos, labels, test_size=(1 - train_ratio), stratify=labels, random_state=42
        )

        relative_test_size = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=42
        )

        logger.info(f"Split Ratios: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        splits = {'train': X_train, 'val': X_val, 'test': X_test}
        
        for split_name, files in splits.items():
            logger.info(f"Moving {len(files)} files to '{split_name}' set...")
            for src_path in files:
                label = src_path.parent.name
                dest_dir = self.output_dir / split_name / label
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, dest_dir / src_path.name)
        
        logger.info("Splitting Completed Successfully.")

    def balance_and_augment(self, target_per_class=None):
        logger.info("========== STEP 2: BALANCING & AUGMENTING (TRAIN ONLY) ==========")
        
        train_dir = self.output_dir / 'train'
        if not train_dir.exists():
            logger.error("Train directory not found. Please run split_dataset first.")
            return

        labels = [d.name for d in train_dir.iterdir() if d.is_dir()]
        
        counts = {lbl: len(list((train_dir / lbl).glob("*.mp4"))) for lbl in labels}
        logger.info(f"Current Class Distribution: {counts}")
        
        if target_per_class is None:
            target_per_class = max(counts.values())
            
        logger.info(f"Target set to: {target_per_class} videos/class")

        augment_methods = [
            ('flip', self.augmentor.apply_flip),
            ('noise', self.augmentor.apply_gaussian_noise),
            ('cutout', self.augmentor.apply_cutout),
            ('speed', self.augmentor.apply_speed),
            ('bright', self.augmentor.apply_brightness)
        ]

        total_generated = 0

        for label in tqdm(labels, desc="Balancing Classes"):
            current_files = list((train_dir / label).glob("*.mp4"))
            current_count = len(current_files)
            needed = target_per_class - current_count
            
            if needed <= 0:
                logger.info(f"Class '{label}' is balanced (Count: {current_count}). Skipping.")
                continue

            logger.info(f"Generating {needed} augmented videos for class '{label}'...")
            
            generated = 0
            while generated < needed:
                try:
                    src_vid = random.choice(current_files)
                    frames, fps = self._read_video(src_vid)
                    if not frames: continue

                    method_name, method_func = random.choice(augment_methods)
                    aug_frames = method_func(frames)
                    
                    save_name = f"{src_vid.stem}_aug_{method_name}_{generated}.mp4"
                    self._save_video(aug_frames, train_dir / label / save_name, fps)
                    
                    generated += 1
                    total_generated += 1
                except Exception as e:
                    logger.error(f"Error augmenting {src_vid}: {e}")

        logger.info(f"Augmentation Finished. Total new videos created: {total_generated}")

        final_counts = {lbl: len(list((train_dir / lbl).glob("*.mp4"))) for lbl in labels}
        logger.info(f"Final Class Distribution: {final_counts}")

def main():
    setup_logger('./logging/augment.log')
    pipeline = PipelineManager(
        source_dir='./data/processed/', 
        output_dir='./data/training_dataset/'
    )
    
    pipeline.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    pipeline.balance_and_augment(target_per_class=None)

if __name__ == "__main__":
    main()