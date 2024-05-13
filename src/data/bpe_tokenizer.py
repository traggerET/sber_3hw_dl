
from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    def __init__(self, sentence_list):
        """
        sentence_list - список предложений для обучения
        """
	self.tok = self.train(sentence_list)
	self.w2i = self.tok.get_vocab()
        self.i2w = {id: w for w, id in self.w2i.items()}

    
    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        return self.tok.encode(sentence).ids


    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        return self.tok.decode(token_list, skip_special_tokens=True).split()

    
    def train(self, sentence_list):
        self.tok = Tokenizer(BPE(unk_token="[UNK]"))
        self.tok.pre_tokenizer = Whitespace()
        self.tok.decoder = decoders.BPEDecoder()

        trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"],
                            end_of_word_suffix="</w>")
        self.tok.train_from_iterator(sentence_list, trainer)

        self.tok.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", self.tokenizer.token_to_id("[SOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ]
        )

        self.max_sent_len = 0
        for s in sentence_list:
            self.max_sent_len = max(self.max_sent_len, len(self(s)))
        self.tok.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), length=self.max_sent_len)
        self.tok.enable_truncation(max_length=self.max_sent_len)

        return self.tok
