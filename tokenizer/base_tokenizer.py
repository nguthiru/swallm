from abc import ABC, abstractmethod


class BaseTokenizer(ABC):

    def __init__(self, model_path: str):
        self.model_path = model_path

    def __call__(self, text: str):
        return self.tokenize(text)

    def tokenize(self, text: str):
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        raise NotImplementedError

    def vocab_size(self):
        raise NotImplementedError
