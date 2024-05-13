import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, max_len):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super(PositionalEncoding, self).__init__()
        
        pos_enc = torch.zeros(max_len, emb_size) 
        poss = torch.arange(0, max_len).unsqueeze(1).float()

        divv = torch.exp(-1 * torch.arange(0, emb_size, 2).float() / emb_size * math.log(10000.0))
        pos_enc[:, 0::2] = torch.sin(poss * divv)
        pos_enc[:, 1::2] = torch.cos(poss * divv)
        pos_enc = pos_enc.unsqueeze(1)

        self.register_buffer("pos_enc", pos_enc)

    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов: (S, B, E)
        """
        return token_embedding + self.pos_enc[:token_embedding.size(0)]
