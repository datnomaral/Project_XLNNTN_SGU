# 🚀 HƯỚNG DẪN CHẠY TRÊN WINDOWS - TỪNG BƯỚC

## ⚠️ QUAN TRỌNG: Python vừa được cài đặt!

Python đã được cài đặt thành công nhưng **chưa được nhận** trong PowerShell hiện tại.

---

## 📝 CÁC BƯỚC THỰC HIỆN:

### **BƯỚC 1: Đóng tất cả cửa sổ PowerShell/Terminal**

Đóng cửa sổ terminal/PowerShell đang mở hiện tại.

---

### **BƯỚC 2: Mở PowerShell MỚI**

**Cách 1 (Khuyến nghị):**
1. Nhấn `Windows + X`
2. Chọn **"Terminal"** hoặc **"Windows PowerShell"**

**Cách 2:**
1. Nhấn `Windows + R`
2. Gõ: `powershell`
3. Enter

---

### **BƯỚC 3: Chuyển đến thư mục dự án**

Trong PowerShell mới, gõ:

```powershell
cd "D:\ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN"
```

Nhấn **Enter**

---

### **BƯỚC 4: Kiểm tra Python**

Gõ:

```powershell
python --version
```

**Nếu thấy:**
```
Python 3.11.9
```
→ ✅ **TUYỆT VỜI!** Chuyển sang BƯỚC 5

**Nếu vẫn lỗi "Python was not found":**
→ Làm theo **PHƯƠNG ÁN DỰ PHÒNG** bên dưới

---

### **BƯỚC 5: Chạy script tự động**

Gõ:

```powershell
.\RUN_LOCAL.bat
```

Hoặc **double-click** vào file `RUN_LOCAL.bat` trong File Explorer

Script sẽ tự động:
- ✅ Tạo virtual environment
- ✅ Cài đặt tất cả thư viện (torch, spacy, nltk,...)
- ✅ Download spaCy models
- ✅ Mở Jupyter Notebook

**Thời gian:** ~10-15 phút

---

### **BƯỚC 6: Chạy Notebook**

Khi Jupyter Notebook mở trong trình duyệt:

1. File `main.ipynb` sẽ tự động mở
2. Trong menu: **Cell → Run All**
3. Hoặc nhấn: **Shift + Enter** từng cell

**Thời gian training:** 30-60 phút (CPU) hoặc 10-20 phút (GPU)

---

### **BƯỚC 7: Kiểm tra kết quả**

Sau khi chạy xong, kiểm tra thư mục:

```
checkpoints/
    └── best_model.pth          ✅ Model đã train

results/
    ├── training_history.json   ✅ Lịch sử loss
    ├── training_history.png    ✅ Biểu đồ loss
    ├── bleu_scores.json        ✅ BLEU scores
    ├── bleu_scores.png         ✅ Biểu đồ BLEU
    └── error_analysis.json     ✅ 5 ví dụ lỗi dịch
```

---

## 🔧 PHƯƠNG ÁN DỰ PHÒNG (Nếu Python vẫn không nhận)

### **Option A: Khởi động lại máy**

Đơn giản nhất: **Khởi động lại máy tính**

Sau đó làm lại từ BƯỚC 2

---

### **Option B: Cài đặt thủ công từng bước**

Nếu không muốn khởi động lại, làm theo các lệnh sau:

#### 1. Tạo virtual environment:
```powershell
python -m venv venv
```

Nếu lỗi, thử:
```powershell
py -m venv venv
```

#### 2. Kích hoạt virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

Nếu lỗi "execution policy", chạy:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Gõ `Y` và Enter, sau đó chạy lại lệnh activate.

#### 3. Cài đặt dependencies:
```powershell
pip install --upgrade pip
pip install torch torchtext numpy pandas spacy nltk matplotlib seaborn jupyter notebook tqdm
```

#### 4. Download spaCy models:
```powershell
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

#### 5. Mở Jupyter Notebook:
```powershell
jupyter notebook main.ipynb
```

---

### **Option C: Dùng Anaconda (Nếu đã cài)**

Nếu bạn có Anaconda:

```powershell
conda create -n nlp python=3.11
conda activate nlp
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
jupyter notebook main.ipynb
```

---

## ⚡ TÓM TẮT NHANH

**Nếu Python đã nhận được (sau khi mở PowerShell mới hoặc restart):**

```powershell
cd "D:\ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN"
.\RUN_LOCAL.bat
```

→ Đợi script chạy → Jupyter mở → Run All → Đợi 30-60 phút → XONG!

---

## 🛠️ Troubleshooting

### ❌ "python : The term 'python' is not recognized"
**Giải pháp:**
1. Mở PowerShell mới (đóng cũ)
2. Hoặc khởi động lại máy
3. Hoặc thử `py` thay vì `python`

### ❌ "execution of scripts is disabled"
**Giải pháp:**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### ❌ "CUDA out of memory" khi train
**Giải pháp:**
Trong notebook, sửa:
```python
BATCH_SIZE = 16  # Giảm từ 64 xuống
HIDDEN_SIZE = 256  # Giảm từ 512 xuống
```

### ❌ "No module named 'torch'"
**Giải pháp:**
```powershell
pip install torch torchtext
```

---

## 📞 CẦN HỖ TRỢ?

Nếu vẫn gặp vấn đề:

1. **Đọc lại file này** - Có thể bạn bỏ qua bước nào đó
2. **Kiểm tra:** `python --version` trong PowerShell MỚI
3. **Google error message** - Hầu hết lỗi đều có trên StackOverflow
4. **Hoặc dùng Google Colab** - Đơn giản hơn nhiều!

---

## ✅ CHECKLIST

Trước khi chạy, đảm bảo:

- [ ] Đã đóng PowerShell cũ và mở mới
- [ ] Đã chuyển đến đúng thư mục dự án
- [ ] `python --version` hiển thị Python 3.11.x
- [ ] Có kết nối internet (để download dataset)
- [ ] Có ít nhất 5GB dung lượng trống

Sau khi chạy xong:

- [ ] Có file `best_model.pth` trong `checkpoints/`
- [ ] Có các biểu đồ PNG trong `results/`
- [ ] BLEU scores hiển thị trong notebook
- [ ] Đã lưu notebook (Ctrl+S)

---

## 🎯 BƯỚC TIẾP THEO

Sau khi chạy thành công:

1. ✅ **Xem kết quả:** Mở các file PNG trong `results/`
2. ✅ **Viết báo cáo:** Theo `report/REPORT_GUIDE.md`
3. ✅ **Nộp bài:** `main.ipynb` + `report.pdf` + `best_model.pth`

**Deadline: 14/12/2025 (23:59)**

---

**CHÚC BẠN THÀNH CÔNG! 🚀**

Nếu có vấn đề gì, hãy chụp màn hình lỗi và đọc phần Troubleshooting!
