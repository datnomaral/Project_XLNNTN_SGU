"""
Data utilities cho dự án dịch máy Anh-Pháp
Không dùng torchtext – load Multi30k từ file local
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy


# =====================================================================
# 1️⃣ Vocabulary
# =====================================================================

class Vocabulary:
    """
    Xây dựng từ điển từ dữ liệu
    Giới hạn: 10,000 từ phổ biến nhất mỗi ngôn ngữ
    """
    def __init__(self, freq_threshold=1, max_size=10000):
        # Token đặc biệt
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        
        # Mapping
        self.itos = {0: self.pad_token, 1: self.unk_token,
                     2: self.sos_token, 3: self.eos_token}
        self.stoi = {self.pad_token: 0, self.unk_token: 1,
                     self.sos_token: 2, self.eos_token: 3}
        
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocab_from_iterator(self, iterator):
        frequencies = Counter()
        
        for sentence in iterator:
            for word in sentence:
                frequencies[word] += 1
        
        most_common = frequencies.most_common(self.max_size - 4)
        
        idx = 4  # Bỏ qua 4 special tokens
        for word, freq in most_common:
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, tokens):
        return [self.stoi.get(token, self.stoi[self.unk_token]) for token in tokens]


def build_vocab_from_iterator(iterator, tokenizer, max_size=10000):
    vocab = Vocabulary(max_size=max_size)
    
    print("Building vocabulary...")
    tokenized = [tokenizer(s) for s in iterator]
    
    vocab.build_vocab_from_iterator(tokenized)
    print(f"Vocabulary size: {len(vocab)}")
    
    return vocab


# =====================================================================
# 2️⃣ Dataset & Dataloader
# =====================================================================

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab,
                 src_tokenizer, tgt_tokenizer):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        
        src_tokens = self.src_tokenizer(src)
        tgt_tokens = self.tgt_tokenizer(tgt)
        
        src_ids = self.src_vocab.numericalize(src_tokens)
        tgt_ids = [self.tgt_vocab.stoi[self.tgt_vocab.sos_token]] + \
                  self.tgt_vocab.numericalize(tgt_tokens) + \
                  [self.tgt_vocab.stoi[self.tgt_vocab.eos_token]]
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_batch])
    sorted_idx = torch.argsort(src_lengths, descending=True)

    src_batch = [src_batch[i] for i in sorted_idx]
    tgt_batch = [tgt_batch[i] for i in sorted_idx]
    src_lengths = src_lengths[sorted_idx]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    return src_padded, src_lengths, tgt_padded


def get_data_loaders(train_data, val_data, test_data,
                     src_vocab, tgt_vocab,
                     src_tokenizer, tgt_tokenizer,
                     batch_size=64, num_workers=0):

    pad_idx = src_vocab.stoi[src_vocab.pad_token]

    train_loader = DataLoader(
        TranslationDataset(*zip(*train_data), src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_idx),
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        TranslationDataset(*zip(*val_data), src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_idx),
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        TranslationDataset(*zip(*test_data), src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_idx),
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


# =====================================================================
# 3️⃣ spaCy Tokenizers
# =====================================================================

def get_tokenizers():
    try:
        spacy_en = spacy.load("en_core_web_sm")
        spacy_fr = spacy.load("fr_core_news_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"])
        spacy_en = spacy.load("en_core_web_sm")
        spacy_fr = spacy.load("fr_core_news_sm")

    def tokenize_en(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    def tokenize_fr(text):
        return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]

    return tokenize_en, tokenize_fr


# =====================================================================
# 4️⃣ Load Multi30k from files (KHÔNG dùng torchtext)
# =====================================================================

def load_data_from_files(base_path="data/multi30k"):
    """
    Load dataset Multi30k từ các file:
    - train.en, train.fr
    - val.en, val.fr
    - test.en, test.fr
    """

    def read(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    train_en = read(f"{base_path}/train.en")
    train_fr = read(f"{base_path}/train.fr")
    val_en   = read(f"{base_path}/val.en")
    val_fr   = read(f"{base_path}/val.fr")
    test_en  = read(f"{base_path}/test.en")
    test_fr  = read(f"{base_path}/test.fr")

    return (
        list(zip(train_en, train_fr)),
        list(zip(val_en, val_fr)),
        list(zip(test_en, test_fr)),
    )
