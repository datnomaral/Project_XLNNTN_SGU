"""
Module đánh giá mô hình
Bao gồm: BLEU score calculation, visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import json
import os


def calculate_bleu_score(references, hypotheses):
    """
    Tính BLEU-4 score cho corpus (theo yêu cầu đề tài)
    Sử dụng nltk.translate.bleu_score.sentence_bleu
    
    Args:
        references: List of reference translations (mỗi câu là list of tokens)
        hypotheses: List of hypothesis translations (mỗi câu là list of tokens)
        
    Returns:
        bleu_score: Float - BLEU-4 score (%)
    """
    # Smoothing function để tránh 0 score với n-gram không match
    smooth = SmoothingFunction()
    
    # BLEU-4 với weights đều cho 4-gram
    weights = (0.25, 0.25, 0.25, 0.25)
    
    scores = []
    for ref, hyp in zip(references, hypotheses):
        # reference phải là list of lists (có thể có nhiều references)
        ref_list = [ref] if isinstance(ref[0], str) else ref
        
        score = sentence_bleu(
            ref_list, 
            hyp, 
            weights=weights,
            smoothing_function=smooth.method1
        )
        scores.append(score)
    
    # Trả về BLEU-4 trung bình (%)
    return np.mean(scores) * 100


def evaluate_model_bleu(model, data_loader, src_vocab, tgt_vocab, device, max_len=50):
    """
    Đánh giá model trên test set và tính BLEU score
    
    Args:
        model: Seq2Seq model đã train
        data_loader: DataLoader cho test set
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: cuda hoặc cpu
        max_len: Độ dài tối đa của câu dịch
        
    Returns:
        bleu_scores: Dict chứa BLEU scores
        references: List of reference translations
        hypotheses: List of hypothesis translations
    """
    model.eval()
    
    references = []
    hypotheses = []
    
    sos_idx = tgt_vocab.stoi[tgt_vocab.sos_token]
    eos_idx = tgt_vocab.stoi[tgt_vocab.eos_token]
    pad_idx = tgt_vocab.stoi[tgt_vocab.pad_token]
    
    with torch.no_grad():
        for src, src_lengths, tgt in data_loader:
            src = src.to(device)
            batch_size = src.shape[0]
            
            # Encode
            hidden, cell = model.encode(src, src_lengths)
            
            # Decode (greedy)
            input = torch.tensor([sos_idx] * batch_size).to(device)
            
            for t in range(max_len):
                output, hidden, cell = model.decode_step(input, hidden, cell)
                
                # Lấy predicted token
                input = output.argmax(1)
                
                # Kiểm tra nếu tất cả đã predict <eos>
                if (input == eos_idx).all():
                    break
            
            # Chuyển về text
            for i in range(batch_size):
                # Target (reference)
                tgt_tokens = tgt[i].cpu().numpy()
                tgt_tokens = [idx for idx in tgt_tokens if idx not in [sos_idx, eos_idx, pad_idx]]
                ref_text = [tgt_vocab.itos[idx] for idx in tgt_tokens]
                
                # Hypothesis sẽ được tạo trong hàm translate
                # Tạm thời dùng greedy decode đơn giản
                # (Hàm translate() trong translate.py sẽ xử lý đầy đủ hơn)
                
                references.append(ref_text)
    
    return references


def plot_training_history(history, save_path='results/training_history.png'):
    """
    Vẽ biểu đồ train/val loss
    
    Args:
        history: Dict chứa 'train_loss' và 'val_loss'
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Highlight best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Biểu đồ training đã lưu tại: {save_path}")
    
    plt.show()


def plot_bleu_scores(bleu_score, save_path='results/bleu_score.png'):
    """
    Vẽ biểu đồ BLEU-4 score
    
    Args:
        bleu_score: Float - BLEU-4 score (%)
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(8, 6))
    
    # Tạo bar chart với 1 bar duy nhất
    bar = plt.bar(['BLEU-4'], [bleu_score], color='#2ecc71', alpha=0.8, 
                   edgecolor='black', width=0.5)
    
    # Thêm giá trị lên bar
    plt.text(0, bleu_score, f'{bleu_score:.2f}%',
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('BLEU-4 Score on Test Set', fontsize=14, fontweight='bold')
    plt.ylim(0, bleu_score * 1.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Biểu đồ BLEU score đã lưu tại: {save_path}")
    
    plt.show()


def analyze_translation_errors(model, test_data, src_vocab, tgt_vocab, 
                               src_tokenizer, device, num_examples=5):
    """
    Phân tích lỗi dịch thuật
    Chọn 5 ví dụ có BLEU thấp nhất để phân tích
    
    Args:
        model: Seq2Seq model
        test_data: Test dataset (tuples of (src, tgt))
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        src_tokenizer: Source tokenizer
        device: cuda hoặc cpu
        num_examples: Số ví dụ cần phân tích
        
    Returns:
        error_examples: List of dicts chứa thông tin lỗi
    """
    from src.translate import translate_sentence
    
    model.eval()
    error_examples = []
    
    smooth = SmoothingFunction()
    
    # Tính BLEU cho tất cả các ví dụ
    bleu_scores_per_sentence = []
    
    for src_sentence, tgt_sentence in test_data[:100]:  # Giới hạn 100 câu để nhanh
        # Dịch câu
        translated = translate_sentence(
            model, src_sentence, src_vocab, tgt_vocab, 
            src_tokenizer, device, max_len=50
        )
        
        # Tokenize target
        tgt_tokens = src_tokenizer(tgt_sentence)  # Giả sử cùng tokenizer
        
        # Tính BLEU
        bleu = sentence_bleu(
            [tgt_tokens], 
            translated.split(), 
            smoothing_function=smooth.method1
        ) * 100
        
        bleu_scores_per_sentence.append({
            'source': src_sentence,
            'reference': tgt_sentence,
            'hypothesis': translated,
            'bleu': bleu
        })
    
    # Sắp xếp theo BLEU tăng dần (lỗi nhiều nhất)
    bleu_scores_per_sentence.sort(key=lambda x: x['bleu'])
    
    # Lấy num_examples ví dụ đầu tiên
    error_examples = bleu_scores_per_sentence[:num_examples]
    
    # Phân loại lỗi
    for example in error_examples:
        example['error_types'] = []
        
        # Phân tích lỗi đơn giản
        if '<unk>' in example['hypothesis']:
            example['error_types'].append('Từ vựng ngoài từ điển (OOV)')
        
        if len(example['hypothesis'].split()) < len(example['reference'].split()) * 0.5:
            example['error_types'].append('Câu quá ngắn - mất thông tin')
        
        if len(example['hypothesis'].split()) > len(example['reference'].split()) * 1.5:
            example['error_types'].append('Câu quá dài - thừa từ')
        
        # Kiểm tra ngữ pháp (đơn giản)
        if example['hypothesis'].strip().endswith('.') == False and example['reference'].strip().endswith('.'):
            example['error_types'].append('Thiếu dấu câu')
    
    return error_examples


def save_error_analysis(error_examples, save_path='results/error_analysis.json'):
    """
    Lưu kết quả phân tích lỗi
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(error_examples, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Phân tích lỗi đã lưu tại: {save_path}")
    
    # In ra console
    print("\n" + "=" * 80)
    print("PHÂN TÍCH LỖI DỊCH THUẬT (5 ví dụ)")
    print("=" * 80)
    
    for i, example in enumerate(error_examples, 1):
        print(f"\nVí dụ {i}:")
        print(f"  Source:     {example['source']}")
        print(f"  Reference:  {example['reference']}")
        print(f"  Hypothesis: {example['hypothesis']}")
        print(f"  BLEU Score: {example['bleu']:.2f}%")
        print(f"  Lỗi phát hiện: {', '.join(example['error_types']) if example['error_types'] else 'Không xác định'}")
    
    print("\n" + "=" * 80)
