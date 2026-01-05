import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=30, transform=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.mode = mode
        
        self.meta_path = self.root_dir.parent / 'class_map.json'

        if self.mode == 'train' or not self.meta_path.exists():
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self._save_metadata()
        else:
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                self.classes = meta['idx_to_class']
                self.class_to_idx = meta['class_to_idx']

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists(): continue 
            for vid_path in class_dir.glob("*.mp4"):
                self.samples.append((str(vid_path), self.class_to_idx[class_name]))

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

    def __len__(self):
        return len(self.samples)

    def _get_indices(self, total_frames):
        """Logic lấy mẫu khung hình thông minh hơn"""
        if self.mode == 'train':
            seg_size = total_frames // self.num_frames
            if seg_size > 0:
                indices = [np.random.randint(i * seg_size, (i + 1) * seg_size) 
                           for i in range(self.num_frames)]
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        return indices

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        indices = self._get_indices(total_frames)
        sampled_frames = []
        last_idx = -1

        for idx in indices:
            if idx != last_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame)
                last_idx = idx
            else:
                if sampled_frames:
                    sampled_frames.append(sampled_frames[-1])
        
        cap.release()
        
        if len(sampled_frames) == 0:
            return None
            
        while len(sampled_frames) < self.num_frames:
            sampled_frames.append(sampled_frames[-1])

        return sampled_frames[:self.num_frames]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_video(path)

        if frames is None or len(frames) == 0:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        
        frames = [Image.fromarray(f) for f in frames]

        if self.transform:
            frames = [self.transform(f) for f in frames]
        video_tensor = torch.stack(frames) # (T, C, H, W)
        return video_tensor, label

def robust_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]) # Trả về tensor 1D rỗng
    
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels)

def get_dataloaders(data_root='./data/training_dataset', batch_size=16, num_frames=30, num_workers=4):
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.RandomHorizontalFlip(p=0.5), # Tăng cường dữ liệu
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    targets = [s[1] for s in train_ds.samples]
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight)
    
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler, # Thay thế shuffle=True bằng sampler
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