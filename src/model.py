"""
Mô hình Encoder-Decoder LSTM cho dịch máy
Context vector cố định (không sử dụng attention)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Encoder LSTM
    Input: Chuỗi token tiếng Anh
    Output: Context vector (h_n, c_n) - trạng thái cuối cùng của LSTM
    """
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        """
        Args:
            input_size: Kích thước vocabulary tiếng Anh
            embedding_dim: Kích thước embedding (256-512)
            hidden_size: Kích thước hidden state (512)
            num_layers: Số layer LSTM (2)
            dropout: Dropout rate (0.3-0.5)
        """
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Không dùng bidirectional theo đề
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lengths):
        """
        Args:
            src: [batch_size, src_len] - padded source sequences
            src_lengths: [batch_size] - độ dài thực của mỗi sequence
            
        Returns:
            outputs: [batch_size, src_len, hidden_size] - tất cả hidden states
            hidden: ([num_layers, batch_size, hidden_size], ...) - trạng thái cuối
        """
        # Embedding: [batch_size, src_len, embedding_dim]
        embedded = self.dropout(self.embedding(src))
        
        # Pack padded sequence để LSTM xử lý hiệu quả hơn
        packed_embedded = pack_padded_sequence(
            embedded, 
            src_lengths.cpu(),  # Phải chuyển về CPU
            batch_first=True,
            enforce_sorted=True  # Batch đã được sort trong collate_fn
        )
        
        # LSTM forward
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack
        # outputs: [batch_size, src_len, hidden_size]
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        # hidden: [num_layers, batch_size, hidden_size]
        # cell: [num_layers, batch_size, hidden_size]
        
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """
    Decoder LSTM
    Input: Token tiếng Pháp (ở mỗi bước) + Context vector từ Encoder
    Output: Xác suất từ tiếp theo trong tiếng Pháp
    """
    def __init__(self, output_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        """
        Args:
            output_size: Kích thước vocabulary tiếng Pháp
            embedding_dim: Kích thước embedding (256-512)
            hidden_size: Kích thước hidden state (512) - phải bằng encoder
            num_layers: Số layer LSTM (2) - phải bằng encoder
            dropout: Dropout rate (0.3-0.5)
        """
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer để dự đoán từ
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        """
        Args:
            input: [batch_size] - token hiện tại (1 token tại 1 thời điểm)
            hidden: [num_layers, batch_size, hidden_size] - hidden state từ bước trước
            cell: [num_layers, batch_size, hidden_size] - cell state từ bước trước
            
        Returns:
            prediction: [batch_size, output_size] - phân phối xác suất từ tiếp theo
            hidden: [num_layers, batch_size, hidden_size] - hidden state mới
            cell: [num_layers, batch_size, hidden_size] - cell state mới
        """
        # input: [batch_size] -> [batch_size, 1]
        input = input.unsqueeze(1)
        
        # Embedding: [batch_size, 1, embedding_dim]
        embedded = self.dropout(self.embedding(input))
        
        # LSTM forward
        # output: [batch_size, 1, hidden_size]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Prediction: [batch_size, 1, hidden_size] -> [batch_size, output_size]
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Mô hình Seq2Seq hoàn chỉnh (Encoder-Decoder)
    Context vector cố định: sử dụng (h_n, c_n) từ encoder làm khởi tạo cho decoder
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Kiểm tra encoder và decoder có cùng hidden_size và num_layers
        assert encoder.hidden_size == decoder.hidden_size, \
            "Encoder và Decoder phải có cùng hidden_size!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder và Decoder phải có cùng num_layers!"
        
    def forward(self, src, src_lengths, target, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch_size, src_len] - source sequences
            src_lengths: [batch_size] - độ dài source
            target: [batch_size, tgt_len] - target sequences (bao gồm <sos> và <eos>)
            teacher_forcing_ratio: Tỷ lệ sử dụng ground truth (0.5)
            
        Returns:
            outputs: [batch_size, tgt_len, output_size] - dự đoán cho mỗi bước thời gian
        """
        batch_size = target.shape[0]
        tgt_len = target.shape[1]
        tgt_vocab_size = self.decoder.output_size
        
        # Tensor để lưu outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encoder forward
        _, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Context vector cố định = (hidden, cell) cuối cùng của encoder
        # hidden: [num_layers, batch_size, hidden_size]
        # cell: [num_layers, batch_size, hidden_size]
        
        # Token đầu tiên của decoder là <sos>
        input = target[:, 0]  # [batch_size]
        
        # Decoder forward (từng bước thời gian)
        for t in range(1, tgt_len):
            # Dự đoán token tiếp theo
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Lưu output
            outputs[:, t, :] = output
            
            # Teacher forcing: Quyết định sử dụng ground truth hay predicted token
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Token có xác suất cao nhất
            top1 = output.argmax(1)
            
            # Nếu teacher forcing: dùng ground truth, ngược lại: dùng predicted
            input = target[:, t] if teacher_force else top1
        
        return outputs
    
    def encode(self, src, src_lengths):
        """
        Chỉ chạy encoder (dùng cho inference)
        """
        with torch.no_grad():
            _, (hidden, cell) = self.encoder(src, src_lengths)
        return hidden, cell
    
    def decode_step(self, input, hidden, cell):
        """
        Một bước decode (dùng cho inference)
        """
        with torch.no_grad():
            output, hidden, cell = self.decoder(input, hidden, cell)
        return output, hidden, cell
