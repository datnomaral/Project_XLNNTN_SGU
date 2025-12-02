# Đồ án Xử lý Ngôn ngữ Tự nhiên
## Dịch máy Anh-Pháp với mô hình Encoder-Decoder LSTM

### Thông tin đồ án
- **Học kỳ**: HK1 / 2025-2026
- **Hạn nộp**: 14/12/2025 (23:59)
- **Nhóm**: Tối đa 2 sinh viên

### Đề tài
Xây dựng mô hình **Encoder-Decoder LSTM** với **context vector cố định** để giải quyết bài toán **dịch máy từ tiếng Anh sang tiếng Pháp**.

### Dataset
- **Multi30K (en-fr)**
- Train: 29,000 cặp câu
- Validation: 1,000 cặp câu
- Test: 1,000 cặp câu

### Cấu trúc thư mục
```
ĐỒ ÁN XỬ LÍ NGÔN NGỮ TỰ NHIÊN/
├── README.md                    # File này
├── requirements.txt             # Các thư viện cần thiết
├── main.ipynb                   # Jupyter Notebook chính để chạy toàn bộ project
├── data/                        # Thư mục chứa dữ liệu
│   └── multi30k/               # Dataset Multi30K
├── src/                         # Thư mục chứa source code
│   ├── data_utils.py           # Xử lý dữ liệu, tokenization, vocab
│   ├── model.py                 # Định nghĩa Encoder, Decoder, Seq2Seq
│   ├── train.py                 # Hàm huấn luyện
│   ├── evaluate.py              # Hàm đánh giá và tính BLEU score
│   └── translate.py             # Hàm dịch câu mới
├── checkpoints/                 # Thư mục lưu model
│   └── best_model.pth          # Model tốt nhất
├── results/                     # Thư mục kết quả
│   ├── training_history.json   # Lịch sử train/val loss
│   ├── bleu_scores.json        # BLEU scores
│   └── error_analysis.json     # Phân tích lỗi
└── report/                      # Báo cáo
    └── report.pdf              # Báo cáo PDF
```

### Yêu cầu kỹ thuật
- **Ngôn ngữ**: Python 3.8+
- **Framework**: PyTorch (≥ 1.13)
- **Không sử dụng**: seq2seq có sẵn, torchtext.legacy, transformers

### Các tính năng chính
1. ✅ Xây dựng mô hình Encoder-Decoder LSTM từ đầu
2. ✅ Context vector cố định (không dùng attention)
3. ✅ Xử lý dữ liệu: tokenization, vocab (10,000 từ), padding/packing
4. ✅ Huấn luyện với early stopping, checkpoint
5. ✅ Greedy decoding và Beam search
6. ✅ Đánh giá BLEU score
7. ✅ Phân tích lỗi dịch thuật + đề xuất cải tiến

### Cách chạy project
1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Download dataset Multi30K:
```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

3. Chạy notebook chính:
```bash
jupyter notebook main.ipynb
```

### Thang điểm (10 điểm)
| Tiêu chí | Điểm |
|----------|------|
| 1. Triển khai mô hình đúng (Encoder-Decoder LSTM) | 3.0 |
| 2. Xử lý dữ liệu, DataLoader, padding/packing | 2.0 |
| 3. Huấn luyện ổn định, có early stopping, lưu checkpoint | 1.5 |
| 4. Hàm translate() hoạt động với câu mới | 1.0 |
| 5. Đánh giá BLEU score + biểu đồ loss | 1.0 |
| 6. Phân tích 5 ví dụ lỗi + đề xuất cải tiến | 1.0 |
| 7. Chất lượng mã nguồn (sạch, có comment, cấu trúc rõ) | 0.5 |
| 8. Báo cáo (đầy đủ, rõ ràng, có biểu đồ, trích dẫn) | 0.5 |
| **Điểm cộng (mở rộng)** | 1.0 |

### Tài liệu tham khảo
- Sutskever et al. (2014). *Sequence to Sequence Learning with Neural Networks*
- PyTorch Documentation: [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- Multi30K Dataset: [https://github.com/multi30k/dataset](https://github.com/multi30k/dataset)

### Lưu ý quan trọng
- ⚠️ **Không sao chép mã** → sẽ bị 0 điểm
- ✅ Mã nguồn phải chạy được từ đầu đến cuối trên Google Colab hoặc máy local
- ✅ Báo cáo PDF phải bao gồm: sơ đồ kiến trúc, biểu đồ train/val loss, BLEU score, 5 ví dụ lỗi + phân tích
- ✅ Checkpoint mô hình (best_model.pth) **bắt buộc nộp**
