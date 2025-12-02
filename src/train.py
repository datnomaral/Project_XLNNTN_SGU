"""
Module huấn luyện mô hình
Bao gồm: training loop, validation, early stopping, checkpoint saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
import os


def train_epoch(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    """
    Huấn luyện một epoch
    
    Args:
        model: Seq2Seq model
        data_loader: DataLoader cho training
        optimizer: Optimizer
        criterion: Loss function
        clip: Gradient clipping value
        teacher_forcing_ratio: Tỷ lệ teacher forcing
        device: cuda hoặc cpu
        
    Returns:
        epoch_loss: Loss trung bình của epoch
    """
    model.train()
    epoch_loss = 0
    
    for batch_idx, (src, src_lengths, tgt) in enumerate(tqdm(data_loader, desc="Training")):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, src_lengths, tgt, teacher_forcing_ratio)
        
        # output: [batch_size, tgt_len, output_size]
        # tgt: [batch_size, tgt_len]
        
        # Reshape để tính loss (bỏ <sos> token)
        output_dim = output.shape[-1]
        output = output[:, 1:, :].contiguous().view(-1, output_dim)
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        # Tính loss
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    """
    Đánh giá model trên validation/test set
    
    Args:
        model: Seq2Seq model
        data_loader: DataLoader cho validation/test
        criterion: Loss function
        device: cuda hoặc cpu
        
    Returns:
        epoch_loss: Loss trung bình
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch_idx, (src, src_lengths, tgt) in enumerate(tqdm(data_loader, desc="Evaluating")):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Forward pass (không teacher forcing khi evaluate)
            output = model(src, src_lengths, tgt, teacher_forcing_ratio=0)
            
            # Reshape để tính loss
            output_dim = output.shape[-1]
            output = output[:, 1:, :].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            # Tính loss
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)


def train(model, train_loader, val_loader, optimizer, criterion, scheduler,
          num_epochs, clip, teacher_forcing_ratio, device, 
          checkpoint_dir='checkpoints', patience=3):
    """
    Huấn luyện model với early stopping
    
    Args:
        model: Seq2Seq model
        train_loader: DataLoader cho training
        val_loader: DataLoader cho validation
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler
        num_epochs: Số epoch tối đa
        clip: Gradient clipping value
        teacher_forcing_ratio: Tỷ lệ teacher forcing
        device: cuda hoặc cpu
        checkpoint_dir: Thư mục lưu checkpoints
        patience: Số epoch chờ trước khi early stopping
        
    Returns:
        history: Dict chứa lịch sử train/val loss
    """
    # Tạo thư mục checkpoint nếu chưa có
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nBắt đầu huấn luyện trên {device}...")
    print(f"Số epochs tối đa: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"Teacher forcing ratio: {teacher_forcing_ratio}")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                clip, teacher_forcing_ratio, device)
        
        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Lưu history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Lưu best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Lưu checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            
            print(f"✓ Model cải thiện! Lưu checkpoint tại epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            print(f"✗ Validation loss không cải thiện ({epochs_no_improve}/{patience})")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n⚠ Early stopping sau {epoch + 1} epochs!")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    print("\n" + "=" * 80)
    print("Hoàn thành huấn luyện!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Lưu history
    history_path = os.path.join('results', 'training_history.json')
    os.makedirs('results', exist_ok=True)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"Lịch sử training đã lưu tại: {history_path}")
    
    return history


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model từ checkpoint
    
    Args:
        model: Seq2Seq model (architecture phải giống)
        checkpoint_path: Đường dẫn tới file checkpoint
        device: cuda hoặc cpu
        
    Returns:
        model: Model đã load weights
        checkpoint: Dict chứa thông tin checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Đã load model từ checkpoint:")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Train loss: {checkpoint['train_loss']:.4f}")
    print(f"  - Val loss: {checkpoint['val_loss']:.4f}")
    
    return model, checkpoint


def count_parameters(model):
    """
    Đếm số lượng parameters có thể train được
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """
    Khởi tạo weights cho model
    Sử dụng Xavier uniform initialization
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
