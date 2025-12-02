"""
Module dịch câu mới
Bao gồm: Greedy decoding, Beam search
"""

import torch


def translate_sentence(model, sentence, src_vocab, tgt_vocab, src_tokenizer, 
                       device, max_len=50, method='greedy', beam_size=3):
    """
    Dịch một câu từ tiếng Anh sang tiếng Pháp
    
    Args:
        model: Seq2Seq model đã train
        sentence: Câu tiếng Anh (string)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        src_tokenizer: Source tokenizer function
        device: cuda hoặc cpu
        max_len: Độ dài tối đa của câu dịch
        method: 'greedy' hoặc 'beam'
        beam_size: Kích thước beam (nếu dùng beam search)
        
    Returns:
        translated_sentence: Câu tiếng Pháp đã dịch (string)
    """
    model.eval()
    
    if method == 'greedy':
        return greedy_decode(model, sentence, src_vocab, tgt_vocab, 
                           src_tokenizer, device, max_len)
    elif method == 'beam':
        return beam_search_decode(model, sentence, src_vocab, tgt_vocab,
                                 src_tokenizer, device, max_len, beam_size)
    else:
        raise ValueError(f"Unknown decoding method: {method}")


def greedy_decode(model, sentence, src_vocab, tgt_vocab, src_tokenizer, 
                  device, max_len=50):
    """
    Greedy decoding: Chọn token có xác suất cao nhất ở mỗi bước
    
    Args:
        model: Seq2Seq model
        sentence: Câu nguồn (string)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        src_tokenizer: Source tokenizer
        device: cuda hoặc cpu
        max_len: Độ dài tối đa
        
    Returns:
        translated: Câu đã dịch (string)
    """
    with torch.no_grad():
        # Tokenize và numericalize
        tokens = src_tokenizer(sentence.lower())
        numericalized = src_vocab.numericalize(tokens)
        
        # Convert to tensor: [1, src_len]
        src_tensor = torch.tensor(numericalized).unsqueeze(0).to(device)
        src_lengths = torch.tensor([len(numericalized)])
        
        # Encode
        hidden, cell = model.encode(src_tensor, src_lengths)
        
        # Decode
        sos_idx = tgt_vocab.stoi[tgt_vocab.sos_token]
        eos_idx = tgt_vocab.stoi[tgt_vocab.eos_token]
        
        input = torch.tensor([sos_idx]).to(device)
        translated_tokens = []
        
        for _ in range(max_len):
            output, hidden, cell = model.decode_step(input, hidden, cell)
            
            # Lấy token có xác suất cao nhất
            predicted_token = output.argmax(1).item()
            
            # Dừng nếu gặp <eos>
            if predicted_token == eos_idx:
                break
            
            translated_tokens.append(predicted_token)
            input = torch.tensor([predicted_token]).to(device)
        
        # Convert indices to words
        translated_words = [tgt_vocab.itos[idx] for idx in translated_tokens]
        translated = ' '.join(translated_words)
        
        return translated


def beam_search_decode(model, sentence, src_vocab, tgt_vocab, src_tokenizer,
                       device, max_len=50, beam_size=3):
    """
    Beam search decoding: Giữ lại top-k hypotheses ở mỗi bước
    
    Args:
        model: Seq2Seq model
        sentence: Câu nguồn (string)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        src_tokenizer: Source tokenizer
        device: cuda hoặc cpu
        max_len: Độ dài tối đa
        beam_size: Số lượng beams (3-5)
        
    Returns:
        translated: Câu đã dịch tốt nhất (string)
    """
    with torch.no_grad():
        # Tokenize và numericalize
        tokens = src_tokenizer(sentence.lower())
        numericalized = src_vocab.numericalize(tokens)
        
        # Convert to tensor
        src_tensor = torch.tensor(numericalized).unsqueeze(0).to(device)
        src_lengths = torch.tensor([len(numericalized)])
        
        # Encode
        hidden, cell = model.encode(src_tensor, src_lengths)
        
        # Special tokens
        sos_idx = tgt_vocab.stoi[tgt_vocab.sos_token]
        eos_idx = tgt_vocab.stoi[tgt_vocab.eos_token]
        
        # Initialize beams: (sequence, score, hidden, cell)
        beams = [(
            [sos_idx],  # sequence
            0.0,        # score (log probability)
            hidden,     # hidden state
            cell        # cell state
        )]
        
        completed_beams = []
        
        for step in range(max_len):
            all_candidates = []
            
            for seq, score, h, c in beams:
                # Nếu sequence đã kết thúc, giữ nguyên
                if seq[-1] == eos_idx:
                    completed_beams.append((seq, score))
                    continue
                
                # Decode một bước
                input = torch.tensor([seq[-1]]).to(device)
                output, new_h, new_c = model.decode_step(input, h, c)
                
                # Log probabilities
                log_probs = torch.log_softmax(output, dim=1).squeeze(0)
                
                # Lấy top-k tokens
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                
                for log_prob, idx in zip(top_log_probs, top_indices):
                    idx = idx.item()
                    new_seq = seq + [idx]
                    new_score = score + log_prob.item()
                    
                    all_candidates.append((new_seq, new_score, new_h, new_c))
            
            # Sắp xếp và chọn top beam_size candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_size]
            
            # Dừng nếu tất cả beams đã hoàn thành
            if len(beams) == 0:
                break
        
        # Thêm các beams còn lại vào completed
        for seq, score, _, _ in beams:
            if seq[-1] != eos_idx:
                seq = seq + [eos_idx]
            completed_beams.append((seq, score))
        
        # Chọn sequence tốt nhất (score cao nhất)
        if completed_beams:
            # Normalize by length để tránh bias về câu ngắn
            best_seq, best_score = max(completed_beams, 
                                      key=lambda x: x[1] / len(x[0]))
        else:
            best_seq = [sos_idx, eos_idx]
        
        # Remove <sos> and <eos>
        translated_tokens = [idx for idx in best_seq 
                           if idx not in [sos_idx, eos_idx]]
        
        # Convert to words
        translated_words = [tgt_vocab.itos[idx] for idx in translated_tokens]
        translated = ' '.join(translated_words)
        
        return translated


def translate(sentence, model=None, src_vocab=None, tgt_vocab=None, 
              src_tokenizer=None, device=None, method='greedy', beam_size=3):
    """
    Hàm translate() theo yêu cầu đề tài
    Wrapper function cho translate_sentence
    
    Args:
        sentence: Câu tiếng Anh (string)
        model: Seq2Seq model (nếu None, sẽ load từ checkpoint)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        src_tokenizer: Source tokenizer
        device: cuda hoặc cpu
        method: 'greedy' hoặc 'beam'
        beam_size: Beam size (nếu dùng beam search)
        
    Returns:
        translated_sentence: Câu tiếng Pháp (string)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Nếu model chưa được truyền vào, load từ checkpoint
    if model is None:
        print("⚠ Model chưa được truyền vào. Cần load từ checkpoint.")
        return None
    
    # Translate
    translated = translate_sentence(
        model, sentence, src_vocab, tgt_vocab, src_tokenizer,
        device, max_len=50, method=method, beam_size=beam_size
    )
    
    return translated


def interactive_translation(model, src_vocab, tgt_vocab, src_tokenizer, device):
    """
    Chế độ dịch tương tác (interactive)
    Người dùng nhập câu tiếng Anh, model dịch ra tiếng Pháp
    """
    print("\n" + "=" * 80)
    print("CHẾ ĐỘ DỊCH TƯƠNG TÁC (Interactive Translation)")
    print("Nhập 'quit' hoặc 'exit' để thoát")
    print("=" * 80 + "\n")
    
    while True:
        sentence = input("English: ").strip()
        
        if sentence.lower() in ['quit', 'exit', 'q']:
            print("Tạm biệt!")
            break
        
        if not sentence:
            continue
        
        # Dịch với greedy decoding
        greedy_translation = translate(
            sentence, model, src_vocab, tgt_vocab, src_tokenizer, device, method='greedy'
        )
        
        # Dịch với beam search
        beam_translation = translate(
            sentence, model, src_vocab, tgt_vocab, src_tokenizer, device, method='beam', beam_size=3
        )
        
        print(f"French (Greedy): {greedy_translation}")
        print(f"French (Beam):   {beam_translation}")
        print()
