"""
Script kiểm tra nội dung file best_model.pth
"""
import torch
import os

checkpoint_path = 'checkpoints/best_model.pth'

# Kiểm tra file tồn tại
if not os.path.exists(checkpoint_path):
    print(f"❌ File không tồn tại: {checkpoint_path}")
    exit(1)

# Load checkpoint
print("Đang load checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*60)
print("THÔNG TIN CHECKPOINT")
print("="*60)

# Hiển thị thông tin
print(f"\n📁 File: {checkpoint_path}")
print(f"📦 Size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")

print(f"\n🔑 Keys trong checkpoint:")
for key in checkpoint.keys():
    print(f"   - {key}")

# Chi tiết từng key
print(f"\n📊 Chi tiết:")
if 'epoch' in checkpoint:
    print(f"   Epoch:      {checkpoint['epoch']}")
if 'train_loss' in checkpoint:
    print(f"   Train Loss: {checkpoint['train_loss']:.4f}")
if 'val_loss' in checkpoint:
    print(f"   Val Loss:   {checkpoint['val_loss']:.4f}")

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"\n🧠 Model State Dict: {len(state_dict)} layers")
    print("   Các layer chính:")
    for i, (name, param) in enumerate(state_dict.items()):
        if i < 10:  # Chỉ hiển thị 10 layer đầu
            print(f"      {name}: {param.shape}")
        elif i == 10:
            print(f"      ... và {len(state_dict) - 10} layers khác")
            break

if 'optimizer_state_dict' in checkpoint:
    print(f"\n⚙️  Optimizer State Dict: Có")

print("\n" + "="*60)
print("✅ FILE CHECKPOINT HỢP LỆ!")
print("="*60)
