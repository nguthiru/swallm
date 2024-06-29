import sentencepiece as spm
from base_tokenizer import BaseTokenizer

class SentencePieceTokenizer(BaseTokenizer):
    """
    Kiswahili Sentence Piece Tokenizer
    """
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)
    
    def tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]
    
    def ids_to_tokens(self, ids):
        return [self.sp.IdToPiece(i) for i in ids]
    def vocab_size(self):
        return len(self.sp)


