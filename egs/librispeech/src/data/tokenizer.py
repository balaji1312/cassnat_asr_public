#!/usr/bin/env python3
# 2022 Ruchao Fan

import sentencepiece as spm

class SPTokenizer(object):
    #uses spm to tokenise words
    def __init__(self, model_path, vocab):
        self.model = model_path
        self.vocab = vocab
        self._build_sentence_piece_processor()

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model)

    def text2tokens(self, text, addsos=False):
        #if word in dict replace with token; else 'unk', and add sos to beg of sentence
        words = self.sp.EncodeAsPieces(text)
        tokens = [self.vocab.word2index[w] if w in self.vocab.word2index else
                    self.vocab.word2index['unk'] for w in words]
        if addsos:
            tokens.insert(0, self.vocab.word2index['sos'])
        return tokens

    def tokens2text(self, tokens):
        #decode line of tokens to original text
        tokens = [ self.vocab.index2word[token] for token in list(tokens)]
        return self.sp.DecodePieces(tokens)


