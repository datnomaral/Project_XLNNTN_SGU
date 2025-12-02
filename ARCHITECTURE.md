# SƠ ĐỒ KIẾN TRÚC MÔ HÌNH ENCODER-DECODER LSTM

## 1. Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                   ENCODER-DECODER LSTM                          │
│                  (Context Vector cố định)                       │
└─────────────────────────────────────────────────────────────────┘

INPUT (English)                                OUTPUT (French)
     │                                              ▲
     │                                              │
     ▼                                              │
┌──────────┐                                  ┌──────────┐
│Tokenize  │                                  │Detokenize│
│+ Numeral │                                  │          │
└────┬─────┘                                  └────▲─────┘
     │                                              │
     │  [1, 45, 12, 89, 3]                   [un, homme, ...]
     │                                              │
     ▼                                              │
┌──────────────────┐                         ┌──────────┐
│   EMBEDDING      │                         │ SOFTMAX  │
│   (256-512 dim)  │                         │          │
└────┬─────────────┘                         └────▲─────┘
     │                                              │
     │  [batch, len, emb_dim]                       │
     │                                              │
     ▼                                              │
┌──────────────────┐                         ┌──────────────────┐
│                  │                         │                  │
│  ENCODER LSTM    │                         │  DECODER LSTM    │
│  (2 layers)      │                         │  (2 layers)      │
│  hidden=512      │                         │  hidden=512      │
│  dropout=0.5     │                         │  dropout=0.5     │
│                  │                         │                  │
│  ┌────────┐      │                         │  ┌────────┐      │
│  │ LSTM 1 │      │                         │  │ LSTM 1 │      │
│  └───┬────┘      │                         │  └───▲────┘      │
│      │           │      CONTEXT VECTOR     │      │           │
│  ┌───▼────┐      │      ───────────────    │  ┌───┴────┐      │
│  │ LSTM 2 │      │      (h_n, c_n)         │  │ LSTM 2 │      │
│  └───┬────┘      │      Fixed!             │  └───▲────┘      │
│      │           │                         │      │           │
└──────┼───────────┘                         └──────┼───────────┘
       │                                            │
       │   (h_n, c_n)                               │
       └────────────────► INIT ►───────────────────┘
                         DECODER
                         HIDDEN STATE


┌────────────────────────────────────────────────────────────────┐
│                      TRAINING PROCESS                          │
└────────────────────────────────────────────────────────────────┘

TEACHER FORCING (ratio = 0.5):

   Ground Truth: <sos> un homme assis <eos>
                   │    │    │     │
   Random(0.5)?    ▼    ▼    ▼     ▼
   ┌───────────┬────┬────┬────┬────┬────┐
   │ Decoder   │<sos│ un │homme│assis│    │
   │ Input     │    │    │     │     │    │
   └───────────┴────┴────┴────┴─────┴────┘
                 │    │    │     │
                 ▼    ▼    ▼     ▼
   ┌───────────┬────┬────┬────┬─────┬────┐
   │ Decoder   │ un │homme│assis│<eos>│    │
   │ Output    │    │     │     │     │    │
   └───────────┴────┴────┴─────┴─────┴────┘

   Loss = CrossEntropyLoss(Output, GroundTruth[1:])
   (Bỏ <sos>, ignore <pad>)
```

---

## 2. Chi tiết từng thành phần

### 2.1. ENCODER

```
Input: "A man sitting on a bench"
   │
   ▼ Tokenize
["a", "man", "sitting", "on", "a", "bench"]
   │
   ▼ Numericalize (vocab)
[45, 123, 567, 89, 45, 234]
   │
   ▼ Embedding Layer (256-512 dim)
[batch, src_len, emb_dim]
   │
   ▼ Pack Padded Sequence
PackedSequence(...)  # Xử lý sequences có độ dài khác nhau
   │
   ▼ LSTM Layer 1
(h_1, c_1) ← LSTM(input, (h_0, c_0))
   │
   ▼ Dropout (0.5)
   │
   ▼ LSTM Layer 2
(h_2, c_2) ← LSTM(h_1, (h_1, c_1))
   │
   ▼ Unpack Padded Sequence
outputs: [batch, src_len, hidden_size]
   │
   ▼ Lấy hidden cuối cùng
(h_n, c_n) = Context Vector
    [num_layers, batch, hidden_size]

CONTEXT VECTOR CỐ ĐỊNH:
- h_n: trạng thái ẩn cuối cùng của encoder
- c_n: cell state cuối cùng
- Không thay đổi trong quá trình decode!
```

### 2.2. DECODER

```
Context Vector (h_n, c_n) từ Encoder
   │
   │ Initialize Decoder Hidden State
   ▼
(h_decoder, c_decoder) ← (h_n, c_n)

Loop for t = 0 to max_len:
   │
   │ Bước 1: Lấy input token
   ▼
   if t == 0:
       input_t = <sos>
   else:
       if random() < teacher_forcing_ratio:
           input_t = ground_truth[t]  # Teacher forcing
       else:
           input_t = predicted_token  # Own prediction
   │
   │ Bước 2: Embedding
   ▼
   embedded = Embedding(input_t)  # [batch, emb_dim]
   │
   │ Bước 3: LSTM Decode
   ▼
   (h_t, c_t) ← LSTM(embedded, (h_{t-1}, c_{t-1}))
   │
   │ Bước 4: Linear + Softmax
   ▼
   logits = Linear(h_t)  # [batch, vocab_size]
   probs = Softmax(logits)
   │
   │ Bước 5: Predict token
   ▼
   predicted_token = argmax(probs)
   │
   │ Bước 6: Kiểm tra <eos>
   ▼
   if predicted_token == <eos>:
       BREAK
   └─── Loop ───┘

Output: Sequence of French words
```

### 2.3. TRAINING LOOP

```
FOR each epoch:
    FOR each batch in train_loader:
        
        1. Get data
        ─────────────
        src, src_lengths, tgt = batch
        
        2. Forward pass
        ─────────────
        output = model(src, src_lengths, tgt, 
                      teacher_forcing_ratio=0.5)
        
        output shape: [batch, tgt_len, vocab_size]
        tgt shape:    [batch, tgt_len]
        
        3. Calculate loss
        ─────────────
        # Reshape (bỏ <sos>)
        output = output[:, 1:, :].reshape(-1, vocab_size)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = CrossEntropyLoss(output, tgt)
        
        4. Backward pass
        ─────────────
        optimizer.zero_grad()
        loss.backward()
        
        5. Gradient clipping
        ─────────────
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        6. Update weights
        ─────────────
        optimizer.step()
    
    7. Validation
    ─────────────
    val_loss = evaluate(model, val_loader)
    
    8. Early stopping check
    ─────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            STOP TRAINING!
```

---

## 3. Hyperparameters Summary

```
┌─────────────────────────┬──────────────┬─────────────────┐
│ Parameter               │ Value        │ Tùy chỉnh       │
├─────────────────────────┼──────────────┼─────────────────┤
│ Vocab size (EN)         │ ≤ 10,000     │ Fixed (đề tài)  │
│ Vocab size (FR)         │ ≤ 10,000     │ Fixed (đề tài)  │
│ Embedding dim           │ 256-512      │ Có thể điều chỉnh│
│ Hidden size             │ 512          │ Đề tài gợi ý    │
│ Num LSTM layers         │ 2            │ Đề tài gợi ý    │
│ Dropout                 │ 0.3-0.5      │ Có thể điều chỉnh│
│ Batch size              │ 32-128       │ Tùy GPU/CPU     │
│ Learning rate           │ 0.001        │ Đề tài: Adam    │
│ Optimizer               │ Adam         │ Fixed (đề tài)  │
│ Scheduler               │ ReduceLR     │ Tùy chọn        │
│ Loss function           │ CrossEntropy │ Fixed           │
│ Teacher forcing ratio   │ 0.5          │ Đề tài gợi ý    │
│ Gradient clip           │ 1.0          │ Tùy chỉnh       │
│ Max epochs              │ 10-20        │ Đề tài gợi ý    │
│ Early stopping patience │ 3            │ Đề tài gợi ý    │
└─────────────────────────┴──────────────┴─────────────────┘
```

---

## 4. Inference (Dịch câu mới)

### 4.1. Greedy Decoding

```
Input: "Hello world"
   │
   ▼ Tokenize + Numericalize
[234, 567]
   │
   ▼ ENCODER
(h_n, c_n)
   │
   ▼ DECODER (greedy)
   
   t=0: input=<sos> → LSTM → Softmax → "bonjour" (max prob)
   t=1: input="bonjour" → LSTM → Softmax → "monde" (max prob)
   t=2: input="monde" → LSTM → Softmax → <eos> (STOP)
   │
   ▼
Output: "bonjour monde"
```

### 4.2. Beam Search (beam_size=3)

```
Input: "Hello world"
   │
   ▼ ENCODER
(h_n, c_n)
   │
   ▼ DECODER (beam search)

t=0: 
  Beams: [(<sos>, score=0)]
  Expand: (<sos> → "bonjour", -0.5)
          (<sos> → "salut", -1.2)
          (<sos> → "coucou", -2.1)
  Keep top-3: [("bonjour", -0.5), ("salut", -1.2), ("coucou", -2.1)]

t=1:
  From "bonjour":
    → "monde" (-0.5 + -0.3 = -0.8)
    → "le" (-0.5 + -1.0 = -1.5)
  From "salut":
    → "monde" (-1.2 + -0.4 = -1.6)
  ...
  Keep top-3: [("bonjour monde", -0.8), 
               ("salut monde", -1.6),
               ("bonjour le", -1.5)]

t=2:
  From "bonjour monde":
    → <eos> (-0.8 + -0.1 = -0.9) ✓ COMPLETE
  ...

BEST: "bonjour monde" (score=-0.9)
```

---

## 5. Đánh giá BLEU Score

```
Reference:  "un homme assis sur un banc"
Hypothesis: "un homme sur un banc"

BLEU-1 (unigram):
  Precision = matched_1grams / total_1grams_in_hyp
            = 5/5 = 1.0 = 100%

BLEU-2 (bigram):
  Bigrams in ref: ["un homme", "homme assis", "assis sur", 
                   "sur un", "un banc"]
  Bigrams in hyp: ["un homme", "homme sur", "sur un", "un banc"]
  Matched: 3/4 = 0.75 = 75%

BLEU-3 (trigram):
  Trigrams in ref: ["un homme assis", "homme assis sur", 
                    "assis sur un", "sur un banc"]
  Trigrams in hyp: ["un homme sur", "homme sur un", "sur un banc"]
  Matched: 1/3 = 0.33 = 33%

BLEU-4 (4-gram):
  ...

Final BLEU-4 = BP × exp(1/4 × Σ log(precision_i))
             ≈ 50% (ví dụ)
```

---

## 6. Vấn đề Context Vector cố định

### ❌ HẠN CHẾ:

```
Long Sentence:
"The quick brown fox jumps over the lazy dog in the garden"
    │
    ▼ ENCODER
(h_n, c_n) ← COMPRESS EVERYTHING INTO FIXED VECTOR!
    │
    ▼ DECODER
Phải nhớ toàn bộ thông tin từ 1 vector cố định
→ Mất thông tin với câu dài!
```

### ✅ GIẢI PHÁP: ATTENTION

```
Encoder outputs: [h_1, h_2, h_3, ..., h_n]
                   │    │    │         │
Decoder at t=3:    │    │    │         │
  "Which encoder   │    │    │         │
   state is most   └────┴────┴─────────┘
   relevant now?"        ▼
                     ATTENTION
                 (dynamic context vector)
```

---

## 7. Tóm tắt luồng dữ liệu

```
RAW DATA (Multi30K)
    ↓
TOKENIZATION (spaCy)
    ↓
VOCABULARY (10,000 words)
    ↓
NUMERICALIZATION
    ↓
DATALOADER (batch, padding, sorting)
    ↓
MODEL (Encoder-Decoder LSTM)
    ↓
TRAINING (Adam, CrossEntropyLoss, Early Stopping)
    ↓
CHECKPOINT (best_model.pth)
    ↓
EVALUATION (BLEU score)
    ↓
INFERENCE (translate new sentences)
```

---

**Tạo file này bằng:**
- **draw.io** (https://app.diagrams.net/)
- **Lucidchart**
- **PowerPoint** (Insert → Shapes → SmartArt)
- **LaTeX TikZ** (cho báo cáo chuyên nghiệp)

Sau đó screenshot và chèn vào báo cáo PDF!
