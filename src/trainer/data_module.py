import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=30, transform=None, mode='train'):
        """
        Args:
            root_dir (str): Đường dẫn đến folder (VD: 'data/dataset/train')
            num_frames (int): Số lượng frame cố định model cần (T).
            transform: Các bước augment/normalize (torchvision.transforms).
            mode (str): 'train', 'val', hoặc 'test'.
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.mode = mode
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for vid_path in class_dir.glob("*.mp4"):
                self.samples.append((str(vid_path), self.class_to_idx[class_name]))

        if self.mode == 'train':
            self._save_metadata()

        print(f"[{mode.upper()}] Loaded {len(self.samples)} videos from {len(self.classes)} classes.")

    def _save_metadata(self):
        meta_path = self.root_dir.parent / 'class_map.json'
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'idx_to_class': self.classes, 
                    'class_to_idx': self.class_to_idx
                }, 
                f, 
                indent=4,
                ensure_ascii=False
            )

    def _load_video(self, path):
        """Đọc video và lấy mẫu T frames"""
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()

        if len(frames) == 0:
            return None

        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        sampled_frames = [frames[i] for i in indices]
        
        return sampled_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        frames = self._load_video(path)

        if frames is None:
            return None
        
        if self.transform:
            frames = [Image.fromarray(f) for f in frames]
            frames = [self.transform(f) for f in frames]
        video_tensor = torch.stack(frames) 
        
        return video_tensor, label

def robust_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]) # Trả về tensor 1D rỗng
    
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels)

def get_dataloaders(data_root='./data/training_dataset', batch_size=16, num_frames=30, num_workers=4):
    """
    Factory function để tạo cả 3 loaders một lúc
    """
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = VideoDataset(f"{data_root}/train", num_frames, train_transform, mode='train')
    val_ds = VideoDataset(f"{data_root}/val", num_frames, val_test_transform, mode='val')
    test_ds = VideoDataset(f"{data_root}/test", num_frames, val_test_transform, mode='test')

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        collate_fn=robust_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        collate_fn=robust_collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        collate_fn=robust_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, len(train_ds.classes)

def example():
    train_dl, val_dl, test_dl, num_classes = get_dataloaders(batch_size=2)
    
    print(f"Num classes: {num_classes}")
    for videos, labels in train_dl:
        print("Video Batch Shape:", videos.shape)(2, 30, 3, 224, 224)
        print("Label Batch Shape:", labels.shape)
        break

if __name__ == "__main__":

    import multiprocessing
    multiprocessing.freeze_support()
    
    print("Running test...")
    example()
# python -m src.data_module 