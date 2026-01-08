import torch
import torch.nn as nn
import os
import json
import numpy as np
from tqdm import tqdm

# --- ONE EPOCH ---
def train_epoch(
    model, loader, criterion, optimizer, 
    device, scaler, accumulation_steps=1, max_norm=1.0
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for i, (videos, labels) in enumerate(pbar):
        if videos.numel() == 0 or len(videos.shape) < 5:
            continue
        videos, labels = videos.to(device), labels.to(device)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step Optimizer
        if (i + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            
            optimizer.zero_grad()

        # Metrics
        running_loss += loss.item() * accumulation_steps * videos.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})

    return running_loss / len(loader.dataset), correct / total

# --- EVALUATE ---
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Evaluating", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader.dataset), correct / total

# --- MAIN TRAIN ---
def train(
    model,
    train_loader,
    eval_loader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=20,
    device='cuda',
    patience=5,
    accumulation_steps=1,
    checkpoints_dir='./checkpoints',
    use_amp=True,
    save_name='best_model.pth'
):
    os.makedirs(checkpoints_dir, exist_ok=True)
    device = torch.device(device)
    model.to(device)
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Start training on {device} | AMP: {use_amp} | Accumulation: {accumulation_steps}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, scaler, accumulation_steps, max_norm=1.0
        )
        
        # Eval
        val_loss, val_acc = evaluate(model, eval_loader, criterion, device)
        
        # Scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save Best & Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            save_path = os.path.join(checkpoints_dir, save_name)
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved Best Model to {save_name} (Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early Stopping triggered at epoch {epoch+1}!")
            break
            
    history_name = f"history_{save_name.replace('.pth', '.json')}"
    history_path = os.path.join(checkpoints_dir, history_name)
    
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f">>> Saved training history to {history_name}")
        
    return history