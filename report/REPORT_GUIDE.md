# Hướng dẫn viết báo cáo PDF - Đồ án Dịch máy Anh-Pháp

## Cấu trúc báo cáo (6-10 trang A4)

### 1. Trang bìa
- Tên trường, khoa
- Đề tài: DỊCH MÁY ANH-PHÁP VỚI MÔ HÌNH ENCODER-DECODER LSTM
- Họ tên sinh viên, MSSV
- Giảng viên hướng dẫn
- Học kỳ: HK1 / 2025-2026
- Ngày nộp: 14/12/2025

---

### 2. Mục lục

---

### 3. Giới thiệu (0.5 trang)
- Bối cảnh: Dịch máy là gì? Tầm quan trọng?
- Mục tiêu: Xây dựng mô hình Encoder-Decoder LSTM để dịch Anh-Pháp
- Phạm vi: Dataset Multi30K, context vector cố định
- Tổ chức báo cáo

---

### 4. Cơ sở lý thuyết (1.5 trang)

#### 4.1. Sequence-to-Sequence Learning
- Giới thiệu Seq2Seq (Sutskever et al., 2014)
- Ứng dụng: dịch máy, chatbot, tóm tắt văn bản

#### 4.2. LSTM (Long Short-Term Memory)
- Vấn đề của RNN: vanishing/exploding gradients
- Cấu trúc LSTM: forget gate, input gate, output gate
- Công thức toán học

#### 4.3. Encoder-Decoder Architecture
- **Encoder**: Đọc câu nguồn → context vector (h_n, c_n)
- **Decoder**: Nhận context vector → sinh câu đích
- **Context vector cố định**: Không dùng attention (baseline)

**Sơ đồ kiến trúc:**
```
Input (English)
     ↓
  Embedding
     ↓
 Encoder LSTM (2 layers, hidden=512)
     ↓
Context Vector (h_n, c_n)
     ↓
 Decoder LSTM (2 layers, hidden=512)
     ↓
  Softmax
     ↓
Output (French)
```

---

### 5. Dữ liệu và tiền xử lý (1 trang)

#### 5.1. Dataset Multi30K (en-fr)
- **Kích thước:**
  - Train: 29,000 cặp câu
  - Validation: 1,000 cặp câu
  - Test: 1,000 cặp câu
- **Đặc điểm:** Câu ngắn (10-15 từ), mô tả hình ảnh

#### 5.2. Tokenization
- Sử dụng **spaCy** (en_core_web_sm, fr_core_news_sm)
- Tách từ, chuyển về lowercase
- **Ví dụ:**
  - Input: "A man sitting on a bench."
  - Tokens: ['a', 'man', 'sitting', 'on', 'a', 'bench', '.']

#### 5.3. Xây dựng từ điển (Vocabulary)
- Giới hạn: **10,000 từ phổ biến nhất** mỗi ngôn ngữ
- Tokens đặc biệt: `<unk>`, `<pad>`, `<sos>`, `<eos>`
- Xử lý từ ngoài từ điển (OOV) → `<unk>`

#### 5.4. Padding & Packing
- **Padding:** Đồng bộ độ dài batch → `pad_sequence()`
- **Packing:** Tối ưu tính toán → `pack_padded_sequence()`
- **Sắp xếp batch:** Theo độ dài giảm dần (`enforce_sorted=True`)

---

### 6. Xây dựng mô hình (2 trang)

#### 6.1. Encoder
- **Input:** Chuỗi token tiếng Anh → Embedding (dim=256)
- **LSTM:** 2 layers, hidden_size=512, dropout=0.5
- **Output:** Context vector (h_n, c_n)

**Công thức:**
```
(h_t, c_t) = LSTM(embed(x_t), (h_{t-1}, c_{t-1}))
```

#### 6.2. Decoder
- **Input:** Token tiếng Pháp ở bước t + context vector
- **LSTM:** 2 layers, hidden_size=512, dropout=0.5
- **Output:** Phân phối xác suất từ tiếp theo

**Công thức:**
```
(h_t, c_t) = LSTM(embed(y_{t-1}), (h_{t-1}, c_{t-1}))
p(y_t) = softmax(Linear(h_t))
```

#### 6.3. Seq2Seq
- Kết nối Encoder và Decoder
- **Teacher forcing ratio:** 0.5 (50% dùng ground truth)

**Bảng tham số:**
| Tham số | Giá trị |
|---------|---------|
| Hidden size | 512 |
| Embedding dim | 256-512 |
| Số layer LSTM | 2 |
| Dropout | 0.3-0.5 |
| Teacher forcing ratio | 0.5 |

---

### 7. Huấn luyện mô hình (1 trang)

#### 7.1. Cấu hình huấn luyện
- **Loss function:** CrossEntropyLoss (ignore padding)
- **Optimizer:** Adam (lr=0.001)
- **Scheduler:** ReduceLROnPlateau (patience=2)
- **Epochs:** 10-20
- **Batch size:** 32-128
- **Early stopping:** patience=3

#### 7.2. Kết quả huấn luyện
**Biểu đồ Train/Val Loss:**
- Chèn hình `results/training_history.png`
- Phân tích: Model hội tụ sau X epochs, val loss thấp nhất = Y

**Checkpoint:**
- Lưu best model tại `checkpoints/best_model.pth`

---

### 8. Đánh giá mô hình (1.5 trang)

#### 8.1. BLEU Score
- **Giới thiệu:** BLEU (Bilingual Evaluation Understudy) đo overlap n-gram
- **Công thức:** BLEU-n = BP × exp(1/n × Σ log(precision_i))
- **Kết quả:**

**Biểu đồ BLEU Scores:**
- Chèn hình `results/bleu_scores.png`

| Metric | Score |
|--------|-------|
| BLEU-1 | XX.XX% |
| BLEU-2 | XX.XX% |
| BLEU-3 | XX.XX% |
| BLEU-4 | XX.XX% |

#### 8.2. Ví dụ dịch
**Dịch tốt:**
```
English:    A man is sitting on a bench.
Reference:  Un homme est assis sur un banc.
Hypothesis: Un homme assis sur un banc.
BLEU:       85.2%
```

**Dịch kém:**
```
English:    The cat is sleeping on the sofa.
Reference:  Le chat dort sur le canapé.
Hypothesis: Le chat <unk> sur le canapé.
BLEU:       42.1%
```

---

### 9. Phân tích lỗi (1 trang)

#### 9.1. 5 ví dụ lỗi dịch (từ `results/error_analysis.json`)

**Ví dụ 1:**
- **Source:** ...
- **Reference:** ...
- **Hypothesis:** ...
- **BLEU:** XX%
- **Lỗi phát hiện:** Từ vựng OOV (Out-of-Vocabulary)

**Ví dụ 2-5:** (Tương tự)

#### 9.2. Phân loại lỗi
| Loại lỗi | Tỷ lệ |
|----------|-------|
| Từ vựng OOV (` <unk>`) | 40% |
| Câu quá ngắn - mất thông tin | 30% |
| Câu quá dài - thừa từ | 20% |
| Thiếu dấu câu | 10% |

---

### 10. Đề xuất cải tiến (1 trang)

#### 10.1. Thêm Attention Mechanism
- **Vấn đề:** Context vector cố định → mất thông tin với câu dài
- **Giải pháp:** Attention động (Bahdanau/Luong)
- **Kỳ vọng:** Cải thiện BLEU +5-10%

#### 10.2. Beam Search
- **Hiện tại:** Greedy decoding (chọn token tốt nhất)
- **Đề xuất:** Beam search (beam size = 3-5) → nhiều hypotheses
- **Kỳ vọng:** BLEU +2-3%

#### 10.3. Subword Tokenization (BPE)
- **Vấn đề:** OOV với từ hiếm
- **Giải pháp:** Byte Pair Encoding → chia từ thành subwords
- **Ví dụ:** "unbelievable" → "un" + "believable"

#### 10.4. Dataset lớn hơn (WMT 2014)
- Multi30K: 29,000 câu
- WMT 2014: ~36 triệu câu
- → Model học được nhiều pattern hơn

#### 10.5. Khác
- Layer normalization
- Scheduled sampling (giảm teacher forcing theo epoch)
- Data augmentation (back-translation)

---

### 11. Kết luận (0.5 trang)
- **Đã làm được:**
  - ✅ Xây dựng Encoder-Decoder LSTM từ đầu
  - ✅ Xử lý dữ liệu Multi30K (tokenization, vocab, padding/packing)
  - ✅ Huấn luyện với early stopping, checkpoint
  - ✅ Đánh giá BLEU score
  - ✅ Phân tích lỗi dịch thuật
  - ✅ Hàm `translate()` hoạt động với câu mới
  
- **Kết quả:**
  - BLEU-4: XX.XX% trên test set
  - Model parameters: ~XX triệu
  
- **Hạn chế:**
  - Context vector cố định → mất thông tin với câu dài
  - Dataset nhỏ → khả năng tổng quát hóa hạn chế
  
- **Hướng phát triển:**
  - Thêm attention, beam search, BPE
  - So sánh với Transformer

---

### 12. Tài liệu tham khảo

1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. In *NIPS* (pp. 3104-3112).

2. Cho, K., Van Merriënboer, B., Gulcehre, C., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *arXiv preprint arXiv:1406.1078*.

3. PyTorch Documentation: torch.nn.LSTM. https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

4. Multi30K Dataset: https://github.com/multi30k/dataset

5. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. In *ACL* (pp. 311-318).

---

### Phụ lục
- Code đầy đủ: `main.ipynb`
- Checkpoint: `checkpoints/best_model.pth`
- Kết quả: `results/*.json`, `results/*.png`

---

## Checklist nộp bài

✅ **Mã nguồn:**
- [ ] `main.ipynb` (Jupyter Notebook) - chạy được từ đầu đến cuối
- [ ] Các file `.py` trong thư mục `src/`
- [ ] Có comment rõ ràng, cấu trúc sạch

✅ **Báo cáo PDF:**
- [ ] Đầy đủ nội dung theo outline trên
- [ ] Có sơ đồ kiến trúc model
- [ ] Có biểu đồ train/val loss
- [ ] Có biểu đồ BLEU scores
- [ ] Có 5 ví dụ lỗi dịch + phân tích
- [ ] Có trích dẫn tài liệu tham khảo

✅ **Checkpoint mô hình:**
- [ ] `best_model.pth` (file .pth)

✅ **Kiểm tra lần cuối:**
- [ ] Notebook chạy được trên Google Colab hoặc máy local
- [ ] Không sao chép code từ nguồn khác
- [ ] Hàm `translate(sentence: str) -> str` hoạt động đúng
- [ ] BLEU score được tính trên test set (trên tập test)
- [ ] Báo cáo PDF xuất ra đẹp, không lỗi font

---

**Lưu ý:**
- Báo cáo nên viết bằng Microsoft Word hoặc LaTeX
- Font: Times New Roman, size 13 (hoặc 12)
- Căn lề: trái phải 2cm, trên dưới 2.5cm
- Hình ảnh phải rõ nét, có caption và số thứ tự
- Trích dẫn theo chuẩn (IEEE, APA, ...)

---

**SUCCESS!** 🎉
