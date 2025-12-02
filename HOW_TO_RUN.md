# 🚀 HƯỚNG DẪN CHẠY DỰ ÁN - NHANH GỌN

## ⚡ CÁCH NHANH NHẤT: 

### **Trên Windows (Vừa cài Python xong):**

1. **Khởi động lại PowerShell hoặc máy tính** (để Python được nhận)

2. **Double-click vào file:** `RUN_LOCAL.bat`

3. Script sẽ tự động:
   - ✅ Tạo virtual environment
   - ✅ Cài đặt tất cả thư viện
   - ✅ Download spaCy models
   - ✅ Mở Jupyter Notebook

4. **Chạy notebook:** Runtime → Run all

✅ **XONG!**

---

## 🌟 HOẶC: Google Colab (Không cần cài Python!)

### **Lý do nên dùng Colab:**
- ✅ **GPU miễn phí** → Chạy nhanh gấp 10 lần
- ✅ **Không cần cài đặt gì** → Chỉ cần trình duyệt
- ✅ **Chạy mọi lúc mọi nơi** → Có internet là được

### **Các bước:**

**1. Upload lên Google Drive:**
- Mở: https://drive.google.com
- Upload toàn bộ thư mục `ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN`

**2. Mở main.ipynb:**
- Click chuột phải → "Open with Google Colaboratory"

**3. Thêm 2 cells đầu:**

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: Chuyển thư mục
%cd /content/drive/MyDrive/ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN
```

**4. Chạy tất cả:**
- Menu: **Runtime → Run all**
- Hoặc: **Ctrl + F9**

**5. Đợi 30-60 phút**

✅ **XONG!** Kết quả trong `checkpoints/` và `results/`

---

## 🎯 Sau khi chạy xong:

### ✅ Kiểm tra các file đã tạo:

```
checkpoints/
    └── best_model.pth           (Model đã train)

results/
    ├── training_history.json    (Lịch sử loss)
    ├── training_history.png     (Biểu đồ loss)
    ├── bleu_scores.json         (BLEU scores)
    ├── bleu_scores.png          (Biểu đồ BLEU)
    └── error_analysis.json      (5 ví dụ lỗi)
```

### ✅ BLEU Scores kỳ vọng:
- **BLEU-1:** ~60-70%
- **BLEU-2:** ~40-50%
- **BLEU-3:** ~25-35%
- **BLEU-4:** ~15-25%

---

## 🛠️ Troubleshooting

### ❌ Lỗi: "Python was not found"
**Giải pháp:**
1. Khởi động lại PowerShell/Terminal
2. Hoặc khởi động lại máy
3. Hoặc dùng Google Colab

### ❌ Lỗi: "CUDA out of memory"
**Giải pháp:**
Trong notebook, sửa:
```python
BATCH_SIZE = 32  # Giảm xuống 16 hoặc 8
HIDDEN_SIZE = 256  # Giảm xuống 256
```

### ❌ Lỗi: "Multi30K dataset download failed"
**Giải pháp:**
1. Download thủ công: https://github.com/multi30k/dataset
2. Đặt vào `data/multi30k/`
3. Uncomment code load local trong notebook

---

## 📋 Checklist

- [ ] Python đã cài (hoặc dùng Colab)
- [ ] Đã chạy `RUN_LOCAL.bat` hoặc upload lên Drive
- [ ] Notebook đã chạy xong
- [ ] Có file `best_model.pth` trong `checkpoints/`
- [ ] Có biểu đồ trong `results/`
- [ ] Đã viết báo cáo PDF (xem `report/REPORT_GUIDE.md`)

---

## ✨ Bước tiếp theo:

1. ✅ **Kiểm tra kết quả:** Xem các biểu đồ và BLEU scores
2. ✅ **Viết báo cáo PDF:** Theo `report/REPORT_GUIDE.md`
3. ✅ **Nộp bài:** `main.ipynb` + `report.pdf` + `best_model.pth`

**Hạn nộp: 14/12/2025 (23:59)**

---

**Chúc bạn thành công! 🎉**
