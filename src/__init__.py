"""
Package src - English to French Machine Translation
Encoder-Decoder LSTM with fixed context vector
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

# Import các modules chính
from .data_utils import (
    Vocabulary,
    TranslationDataset,
    get_tokenizers,
    build_vocab_from_iterator,
    get_data_loaders
)

from .model import (
    Encoder,
    Decoder,
    Seq2Seq
)

from .train import (
    train,
    train_epoch,
    evaluate,
    load_checkpoint,
    count_parameters,
    initialize_weights
)

from .evaluate import (
    calculate_bleu_score,
    plot_training_history,
    plot_bleu_scores,
    analyze_translation_errors,
    save_error_analysis
)

from .translate import (
    translate,
    translate_sentence,
    greedy_decode,
    beam_search_decode,
    interactive_translation
)

__all__ = [
    # Data utils
    'Vocabulary',
    'TranslationDataset',
    'get_tokenizers',
    'build_vocab_from_iterator',
    'get_data_loaders',
    
    # Model
    'Encoder',
    'Decoder',
    'Seq2Seq',
    
    # Training
    'train',
    'train_epoch',
    'evaluate',
    'load_checkpoint',
    'count_parameters',
    'initialize_weights',
    
    # Evaluation
    'calculate_bleu_score',
    'plot_training_history',
    'plot_bleu_scores',
    'analyze_translation_errors',
    'save_error_analysis',
    
    # Translation
    'translate',
    'translate_sentence',
    'greedy_decode',
    'beam_search_decode',
    'interactive_translation',
]
