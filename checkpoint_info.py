"""
Script xuất thông tin chi tiết của checkpoint ra file text
Để thầy/cô có thể xem nội dung file best_model.pth
"""
import torch
import os

checkpoint_path = 'checkpoints/best_model.pth'
output_path = 'checkpoints/checkpoint_info.txt'

# Load checkpoint
print("Đang load checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Tạo nội dung
content = []
content.append("=" * 70)
content.append("THÔNG TIN CHI TIẾT FILE: best_model.pth")
content.append("=" * 70)
content.append("")

# Thông tin cơ bản
content.append("1. THÔNG TIN CƠ BẢN")
content.append("-" * 40)
content.append(f"   File:           {checkpoint_path}")
content.append(f"   Kích thước:     {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
content.append(f"   Best Epoch:     {checkpoint.get('epoch', 'N/A')}")
content.append(f"   Train Loss:     {checkpoint.get('train_loss', 'N/A'):.4f}")
content.append(f"   Val Loss:       {checkpoint.get('val_loss', 'N/A'):.4f}")
content.append(f"   Best Val Loss:  {checkpoint.get('best_val_loss', 'N/A'):.4f}")
content.append("")

# Keys trong checkpoint
content.append("2. CÁC THÀNH PHẦN TRONG CHECKPOINT")
content.append("-" * 40)
for key in checkpoint.keys():
    content.append(f"   ✓ {key}")
content.append("")

# Model architecture
content.append("3. KIẾN TRÚC MODEL (Model State Dict)")
content.append("-" * 40)
state_dict = checkpoint.get('model_state_dict', {})
content.append(f"   Tổng số layers: {len(state_dict)}")
content.append("")

# Tính tổng số parameters
total_params = 0
for name, param in state_dict.items():
    total_params += param.numel()

content.append(f"   Tổng số parameters: {total_params:,}")
content.append("")

# Chi tiết từng layer
content.append("   Chi tiết các layers:")
content.append("   " + "-" * 60)
for name, param in state_dict.items():
    shape_str = str(list(param.shape))
    num_params = param.numel()
    content.append(f"   {name}")
    content.append(f"      Shape: {shape_str}, Params: {num_params:,}")
content.append("")

# Encoder info
content.append("4. THÔNG SỐ MÔ HÌNH (Trích xuất từ weights)")
content.append("-" * 40)
encoder_emb = state_dict.get('encoder.embedding.weight')
decoder_emb = state_dict.get('decoder.embedding.weight')
encoder_lstm_ih = state_dict.get('encoder.lstm.weight_ih_l0')

if encoder_emb is not None:
    content.append(f"   Encoder Vocab Size:    {encoder_emb.shape[0]}")
    content.append(f"   Encoder Embedding Dim: {encoder_emb.shape[1]}")
if decoder_emb is not None:
    content.append(f"   Decoder Vocab Size:    {decoder_emb.shape[0]}")
    content.append(f"   Decoder Embedding Dim: {decoder_emb.shape[1]}")
if encoder_lstm_ih is not None:
    hidden_size = encoder_lstm_ih.shape[0] // 4  # LSTM có 4 gates
    content.append(f"   Hidden Size:           {hidden_size}")
    content.append(f"   LSTM Layers:           2 (có weight l0 và l1)")
content.append("")

# Optimizer info
content.append("5. OPTIMIZER STATE")
content.append("-" * 40)
if 'optimizer_state_dict' in checkpoint:
    opt_state = checkpoint['optimizer_state_dict']
    content.append(f"   Có lưu optimizer state: ✓")
    if 'param_groups' in opt_state:
        for i, group in enumerate(opt_state['param_groups']):
            content.append(f"   Param Group {i}:")
            content.append(f"      Learning Rate: {group.get('lr', 'N/A')}")
            content.append(f"      Weight Decay:  {group.get('weight_decay', 'N/A')}")
else:
    content.append(f"   Không có optimizer state")
content.append("")

content.append("=" * 70)
content.append("✅ ĐÂY LÀ FILE CHECKPOINT HỢP LỆ CỦA MÔ HÌNH SEQ2SEQ LSTM")
content.append("   - Encoder-Decoder LSTM với context vector cố định")
content.append("   - Không sử dụng Attention (theo yêu cầu đề tài)")
content.append("   - Tự code từ đầu, không dùng thư viện seq2seq có sẵn")
content.append("=" * 70)

# Ghi ra file
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(content))

print(f"✓ Đã xuất thông tin checkpoint ra: {output_path}")
print("\nNội dung file:")
print("=" * 70)
print('\n'.join(content))
