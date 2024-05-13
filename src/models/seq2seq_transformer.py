import torch

import metrics
import math

import torch
import torch.nn as nn
from src.models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, encoder_vocab_size: int, decoder_vocab_size: int, dim_feedforward: int, lr: float, device: str, target_tokenizer,
            T, positional_embedding_size=256, n_heads_attention=8, n_encoders=6, n_decoders=6, dropout=0.1,
            final_div_factor=10e+4, start_symbol="[SOS]"):
        super(Seq2SeqTransformer, self).__init__()
        self.device = device
        
        self.transformer = nn.Transformer(d_model=positional_embedding_size, dim_feedforward=dim_feedforward,
            nhead=n_heads_attention, num_encoder_layers=n_encoders, num_decoder_layers=n_decoders, dropout=dropout).to(self.device)

        self.enc_emb = nn.Embedding(encoder_vocab_size, positional_embedding_size).to(self.device)
        self.dec_emb = nn.Embedding(decoder_vocab_size, positional_embedding_size).to(self.device)
        self.positional_encoder = PositionalEncoding(emb_size=positional_embedding_size, max_len=target_tokenizer.max_sent_len).to(self.device)
        self.vocab_projection = nn.Linear(positional_embedding_size, decoder_vocab_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, total_steps=T, max_lr=lr, pct_start=0.1, 
        anneal_strategy='linear', final_div_factor=final_div_factor)
        self.target_tokenizer = target_tokenizer
        self.positional_embedding_size = positional_embedding_size
        self.src_mask = None
        self.trg_mask = None
        self.start_symbol = start_symbol
        self.enc_emb.weight.data.uniform_(-0.1, 0.1)
        self.dec_emb.weight.data.uniform_(-0.1, 0.1)
        self.vocab_projection.bias.data.zero_()
        self.vocab_projection.weight.data.uniform_(-0.1, 0.1)

    def generate_square_subsequent_mask(self, length: int):
        mask = torch.tril(torch.ones((length, length), device=self.device)).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        return mask

    def forward(self, input_tensor: torch.Tensor):
        src = self.positional_encoder(self.enc_emb(input_tensor.transpose(0, 1)))
        memory = self.transformer.encoder(src)
        pred_tokens = [torch.full((input_tensor.size(0),), self.target_tokenizer.word2index[self.start_symbol])]
        each_step_distributions = [nn.functional.one_hot(pred_tokens[0],
                                                         self.vocab_projection.out_features).to(self.device).float()]
        each_step_distributions[0] = each_step_distributions[0].masked_fill(
            each_step_distributions[0] == 0, float('-inf')).masked_fill(each_step_distributions[0] == 1, float(0))
        prediction = torch.full((1, input_tensor.size(0)), self.target_tokenizer.word2index[self.start_symbol], dtype=torch.long, device=self.device)
        for i in range(self.target_tokenizer.max_sent_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(prediction.size(0))
            out = self.transformer.decoder(self.dec_emb(prediction), memory, tgt_mask)
            logits = self.vocab_projection(out[-1])
            _, next_word = torch.max(logits, dim=1)
            prediction = torch.cat([prediction, next_word.unsqueeze(0)], dim=0)
            pred_tokens.append(next_word.clone().detach().cpu())
            each_step_distributions.append(logits)

        return pred_tokens, each_step_distributions

    def training_step(self, batch):
        self.optimizer.zero_grad()
        inp_t, tar_t = batch 
        pred, d_out = self.forward(inp_t)
        tar_t = tar_t[:, :, None]
        tar_l = tar_t.shape[1]
        loss = 0.0
        for di in range(tar_l):
            loss += self.criterion(d_out[di].squeeze(), tar_t[:, di, :].squeeze())
        loss = loss / tar_l
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validation_step(self, batch):
        inp_t, tar_t = batch
        pred, d_out = self.forward(inp_t)
        tar_t = tar_t[:, :, None]
        with torch.no_grad():
            tar_l = tar_t.shape[1]
            loss = 0
            for di in range(tar_l):
                loss += self.criterion(d_out[di].squeeze(), tar_t[:, di, :].squeeze())
            loss = loss / tar_l

        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences


