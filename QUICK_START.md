# HƯỚNG DẪN SỬ DỤNG - ĐỒ ÁN DỊCH MÁY ANH-PHÁP

## 🎯 Mục tiêu dự án
Xây dựng mô hình **Encoder-Decoder LSTM** với **context vector cố định** để dịch từ **tiếng Anh sang tiếng Pháp**.

---

## 📁 Cấu trúc dự án

```
ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN/
├── README.md                    # Tổng quan dự án
├── QUICK_START.md               # File này - Hướng dẫn nhanh
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
│
├── main.ipynb                   # ⭐ NOTEBOOK CHÍNH - Chạy từ đầu đến cuối
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_utils.py           # Xử lý dữ liệu, vocab, DataLoader
│   ├── model.py                 # Encoder, Decoder, Seq2Seq
│   ├── train.py                 # Training loop, early stopping
│   ├── evaluate.py              # BLEU score, visualization
│   └── translate.py             # Greedy/Beam decoding
│
├── data/                        # (Sẽ tự động tạo khi download dataset)
│   └── multi30k/
│
├── checkpoints/                 # (Tự động tạo khi training)
│   └── best_model.pth          # ⭐ Model tốt nhất (NỘP BÀI)
│
├── results/                     # (Tự động tạo khi chạy)
│   ├── training_history.json   # Lịch sử train/val loss
│   ├── training_history.png    # Biểu đồ loss
│   ├── bleu_scores.json        # BLEU scores
│   ├── bleu_scores.png         # Biểu đồ BLEU
│   └── error_analysis.json     # Phân tích lỗi
│
└── report/                      # Báo cáo
    ├── REPORT_GUIDE.md         # Hướng dẫn viết báo cáo PDF
    └── report.pdf              # ⭐ BÁO CÁO CUỐI CÙNG (NỘP BÀI)
```

---

## 🚀 CÁCH CHẠY DỰ ÁN

### Phương pháp 1: Chạy trên Google Colab (Khuyến nghị)

1. **Upload toàn bộ thư mục lên Google Drive**
   ```
   My Drive/ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN/
   ```

2. **Mở `main.ipynb` bằng Google Colab**
   - Click chuột phải vào file → "Open with Google Colaboratory"

3. **Mount Google Drive** (thêm cell đầu tiên):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN
   ```

4. **Chạy toàn bộ notebook** (Runtime → Run all)
   - Thời gian chạy: ~30-60 phút (tùy GPU)

---

### Phương pháp 2: Chạy trên máy local (Windows)

#### Bước 1: Cài đặt Python
- Yêu cầu: Python 3.8 trở lên
- Download: https://www.python.org/downloads/

#### Bước 2: Tạo virtual environment (khuyến nghị)
```powershell
cd "D:\ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN"
python -m venv venv
.\venv\Scripts\activate
```

#### Bước 3: Cài đặt dependencies
```powershell
pip install -r requirements.txt
```

#### Bước 4: Download spaCy models
```powershell
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

#### Bước 5: Chạy Jupyter Notebook
```powershell
jupyter notebook main.ipynb
```

---

## 📊 Kết quả kỳ vọng

### Sau khi chạy xong `main.ipynb`:

✅ **Checkpoints:**
- `checkpoints/best_model.pth` (~XX MB)

✅ **Results:**
- `results/training_history.json` - Lịch sử loss
- `results/training_history.png` - Biểu đồ train/val loss
- `results/bleu_scores.json` - BLEU-1, BLEU-2, BLEU-3, BLEU-4
- `results/bleu_scores.png` - Biểu đồ BLEU
- `results/error_analysis.json` - 5 ví dụ lỗi dịch + phân loại

✅ **Console output:**
- Model architecture
- Training progress (progress bars)
- Best validation loss
- BLEU scores
- 5 translation examples

---

## 🎓 THANG ĐIỂM (10 điểm)

| Tiêu chí | Điểm | Kiểm tra |
|----------|------|----------|
| ✅ Triển khai Encoder-Decoder LSTM đúng | 3.0 | `src/model.py` |
| ✅ Xử lý dữ liệu, DataLoader, padding/packing | 2.0 | `src/data_utils.py` |
| ✅ Huấn luyện ổn định, early stopping, checkpoint | 1.5 | `src/train.py` + `checkpoints/` |
| ✅ Hàm `translate()` hoạt động với câu mới | 1.0 | `src/translate.py` |
| ✅ Đánh giá BLEU score + biểu đồ loss | 1.0 | `src/evaluate.py` + `results/` |
| ✅ Phân tích 5 ví dụ lỗi + đề xuất | 1.0 | `results/error_analysis.json` |
| ✅ Chất lượng code (sạch, comment, cấu trúc) | 0.5 | Toàn bộ `src/` |
| ✅ Báo cáo PDF đầy đủ | 0.5 | `report/report.pdf` |
| 🌟 **Điểm cộng (mở rộng)** | 1.0 | Attention, Beam search, BPE, ... |

---

## 📝 CHECKLIST NỘP BÀI (Hạn: 14/12/2025 23:59)

### ⭐ BẮT BUỘC NỘP:

- [ ] **1. Mã nguồn (Jupyter Notebook hoặc .py)**
  - `main.ipynb` (chạy được từ đầu đến cuối)
  - Hoặc: Toàn bộ thư mục `src/` + `main.ipynb`
  - Có comment rõ ràng

- [ ] **2. Báo cáo PDF**
  - File: `report.pdf`
  - Nội dung: Sơ đồ kiến trúc, biểu đồ, BLEU score, 5 ví dụ lỗi, đề xuất cải tiến
  - Trích dẫn tài liệu tham khảo

- [ ] **3. Checkpoint mô hình**
  - File: `best_model.pth` (hoặc .pt, .ckpt)
  - Phải load được và chạy inference

### ✅ KIỂM TRA CUỐI:

- [ ] Notebook chạy được trên Google Colab hoặc máy local
- [ ] Không sao chép code (tự viết hoặc hiểu rõ)
- [ ] Hàm `translate(sentence: str) -> str` hoạt động đúng
- [ ] BLEU score được tính trên **test set** (không phải train/val)
- [ ] Báo cáo có đầy đủ biểu đồ và phân tích

---

## 🛠️ TROUBLESHOOTING (Xử lý lỗi thường gặp)

### ❌ Lỗi: "No module named 'torchtext'"
```powershell
pip install torchtext
```

### ❌ Lỗi: "Can't find model 'en_core_web_sm'"
```powershell
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

### ❌ Lỗi: "CUDA out of memory"
**Giải pháp:**
1. Giảm batch size: `BATCH_SIZE = 32` hoặc `16`
2. Giảm hidden size: `HIDDEN_SIZE = 256`
3. Chạy trên CPU: `device = torch.device('cpu')`

### ❌ Lỗi: "Multi30K dataset download failed"
**Giải pháp:**
1. Download thủ công từ: https://github.com/multi30k/dataset
2. Đặt vào thư mục `data/multi30k/`
3. Uncomment code load từ file local trong notebook

### ❌ Model không hội tụ (loss không giảm)
**Kiểm tra:**
1. Learning rate quá lớn/nhỏ → Thử `lr=0.0001` hoặc `0.001`
2. Gradient exploding → Kiểm tra `clip=1.0`
3. Data preprocessing sai → In ra vài sample kiểm tra

---

## 📚 TÀI LIỆU THAM KHẢO

### Papers:
1. **Sutskever et al. (2014)** - Sequence to Sequence Learning with Neural Networks
   - https://arxiv.org/abs/1409.3215

2. **Cho et al. (2014)** - Learning Phrase Representations using RNN Encoder-Decoder
   - https://arxiv.org/abs/1406.1078

3. **Bahdanau et al. (2014)** - Neural Machine Translation by Jointly Learning to Align and Translate (Attention)
   - https://arxiv.org/abs/1409.0473

### Documentation:
- PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- torchtext: https://pytorch.org/text/stable/index.html
- spaCy: https://spacy.io/usage
- NLTK BLEU: https://www.nltk.org/api/nltk.translate.html

### Dataset:
- Multi30K: https://github.com/multi30k/dataset
- WMT 2014: http://www.statmt.org/wmt14/translation-task.html

---

## 💡 MẸO ĐẠT ĐIỂM CAO

### 1. Code chất lượng (0.5 điểm)
- ✅ Comment đầy đủ, rõ ràng
- ✅ Đặt tên biến có ý nghĩa
- ✅ Tách module rõ ràng (data, model, train, evaluate)
- ✅ Có docstring cho functions/classes

### 2. Báo cáo chuyên nghiệp (0.5 điểm)
- ✅ Sơ đồ kiến trúc đẹp (vẽ bằng draw.io, PowerPoint)
- ✅ Biểu đồ rõ nét, có caption
- ✅ Phân tích sâu sắc (không chỉ mô tả)
- ✅ Trích dẫn chuẩn (IEEE, APA)

### 3. Điểm cộng (1.0 điểm)
**Lựa chọn 1-2 trong các cải tiến sau:**

#### 🌟 Thêm Attention Mechanism (+0.5 điểm)
- Bahdanau attention hoặc Luong attention
- So sánh với baseline (no attention)

#### 🌟 Beam Search (+0.3 điểm)
- Implement beam search với beam_size = 3, 5, 10
- So sánh BLEU với greedy decoding

#### 🌟 Subword Tokenization (BPE) (+0.3 điểm)
- Sử dụng `sentencepiece` hoặc `subword-nmt`
- Giảm OOV, cải thiện BLEU

#### 🌟 Dataset lớn hơn (WMT 2014) (+0.4 điểm)
- Train trên ~1 triệu câu
- So sánh với Multi30K

---

## 🎯 LUỒNG CÔNG VIỆC KHUYẾN NGHỊ

### Tuần 1-2: Chuẩn bị
- [ ] Đọc hiểu đề tài
- [ ] Nghiên cứu Encoder-Decoder LSTM
- [ ] Thiết lập môi trường (Python, PyTorch, spaCy)

### Tuần 3-4: Coding
- [ ] Viết `data_utils.py` → Test với vài samples
- [ ] Viết `model.py` → Kiểm tra forward pass
- [ ] Viết `train.py` → Chạy 1-2 epochs thử nghiệm
- [ ] Viết `evaluate.py` và `translate.py`

### Tuần 5: Training & Evaluation
- [ ] Train mô hình hoàn chỉnh (10-20 epochs)
- [ ] Đánh giá BLEU score
- [ ] Phân tích lỗi
- [ ] Test hàm translate()

### Tuần 6: Báo cáo
- [ ] Vẽ sơ đồ kiến trúc
- [ ] Viết báo cáo PDF theo template
- [ ] Kiểm tra lại code, checkpoint
- [ ] Nộp bài trước deadline

---

## 📧 HỖ TRỢ

Nếu gặp vấn đề:
1. **Đọc lại REPORT_GUIDE.md** - Có hướng dẫn chi tiết
2. **Google error message** - Hầu hết lỗi PyTorch đã có trên StackOverflow
3. **Hỏi giảng viên** - Email hoặc trong giờ lab

---

## ✨ CHÚC BẠN THÀNH CÔNG!

**Hạn nộp: 14/12/2025 (23:59)**

Nhớ kiểm tra kỹ trước khi nộp:
- ✅ Notebook chạy được
- ✅ Checkpoint tồn tại và load được
- ✅ Báo cáo PDF đầy đủ
- ✅ Code sạch, có comment

**Good luck!** 🚀
