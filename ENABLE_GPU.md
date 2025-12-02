# 🚀 HƯỚNG DẪN SỬ DỤNG GPU CHO TRAINING

## ✅ Bạn đã có GPU RTX 4060 - 8GB VRAM!

---

## BƯỚC 1: Kiểm tra PyTorch có hỗ trợ CUDA không

Trong terminal với `.venv` activated, chạy:

```bash
python check_gpu.py
```

Hoặc chạy trực tiếp:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Kết quả:

#### ✅ Nếu hiện `CUDA available: True`
- **Bạn đã sẵn sàng!** 🎉
- Code sẽ TỰ ĐỘNG dùng GPU
- Chạy training ngay!

#### ❌ Nếu hiện `CUDA available: False`  
- PyTorch đang ở phiên bản CPU-only
- **Cần cài lại PyTorch với CUDA** → Xem BƯỚC 2

---

## BƯỚC 2: Cài lại PyTorch với CUDA (nếu cần)

### 2.1. Activate virtual environment

```bash
.venv\Scripts\activate
```

### 2.2. Gỡ PyTorch hiện tại

```bash
pip uninstall -y torch torchvision torchaudio torchtext
```

### 2.3. Cài PyTorch với CUDA 12.1 (khuyến nghị cho RTX 4060)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Lưu ý**: CUDA 12.1 tương thích với driver CUDA 12.8 của bạn

### 2.4. Cài lại các dependencies khác

```bash
pip install spacy nltk matplotlib seaborn tqdm
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

---

## BƯỚC 3: Verify GPU đang hoạt động

### 3.1. Kiểm tra lại:

```bash
python check_gpu.py
```

Phải thấy:
```
✅ Bạn có thể sử dụng GPU!
GPU name: NVIDIA GeForce RTX 4060 Laptop GPU
```

### 3.2. Trong Jupyter Notebook

Sau khi restart kernel, chạy cell này:

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
```

Output phải là:
```
Device: cuda
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
CUDA Version: 12.1
```

---

## BƯỚC 4: Bắt đầu Training trên GPU

Sau khi verify GPU hoạt động:

1. **Restart Jupyter Kernel**: Kernel → Restart Kernel
2. **Chạy lại từ đầu notebook**
3. **Model sẽ tự động chạy trên GPU** vì code đã có:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = Seq2Seq(...).to(device)  # ← Model lên GPU
   ```

---

## 📊 So sánh tốc độ

| Device | Tốc độ/iteration | Training time (20 epochs) |
|--------|------------------|---------------------------|
| **CPU** | ~1-2s | ~6-10 giờ 😴 |
| **RTX 4060** | ~0.05-0.1s | **20-40 phút** 🚀 |

→ **Nhanh hơn 10-20 lần!**

---

## 🔍 Theo dõi GPU Usage

### Cách 1: Task Manager
- Mở **Task Manager** (Ctrl + Shift + Esc)
- Tab **Performance** → **GPU**
- Xem GPU Usage tăng lên khi training

### Cách 2: nvidia-smi (Real-time)

Terminal mới, chạy:
```bash
nvidia-smi -l 2
```
(Refresh mỗi 2 giây)

---

## ⚠️ Lưu ý

1. **VRAM**: RTX 4060 có 8GB → đủ cho project này
2. **Batch size**: Có thể tăng từ 64 → 128 khi dùng GPU
3. **Overfitting**: Training nhanh hơn → dễ overfit → theo dõi validation loss

---

## 🐛 Troubleshooting

### Lỗi: `CUDA out of memory`
→ Giảm batch size: `BATCH_SIZE = 32`

### Training vẫn chậm trên GPU
→ Check `num_workers` trong DataLoader:
```python
num_workers=2  # Thay vì 0
```

### GPU Usage = 0%
→ Model không lên GPU, check:
```python
print(next(model.parameters()).device)  # Phải là 'cuda:0'
```

---

## 🎯 Tóm tắt Quick Commands

```bash
# 1. Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 2. Nếu False, reinstall PyTorch
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Verify
python check_gpu.py

# 4. Start training in Jupyter!
```

---

**Chúc bạn training thành công! 🚀**
