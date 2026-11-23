"""Transformer encoder and classifier, ported from mvts_transformer src/models/ts_transformer.py.

Parameter names match mvts. Everything except the output head lives outside the
`output_*` names, so pretrained encoder weights load into the classifier by dropping
the keys that start with "output_".
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise ValueError(f"activation should be relu/gelu, not {activation}")


class FixedPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. Input and output: (seq_len, batch, d_model)."""

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", scale_factor * pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :])


class LearnablePositionalEncoding(nn.Module):
    """One learned vector per position. Input and output: (seq_len, batch, d_model)."""

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :])


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    if pos_encoding == "fixed":
        return FixedPositionalEncoding
    raise NotImplementedError(f"pos_encoding should be 'learnable'/'fixed', not '{pos_encoding}'")


class TransformerBatchNormEncoderLayer(nn.Module):
    """nn.TransformerEncoderLayer with BatchNorm instead of LayerNorm (post-norm).

    BatchNorm statistics cover every position of the batch, padding included.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """src: (seq_len, batch, d_model)."""

        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src.permute(1, 2, 0)).permute(2, 0, 1)  # BatchNorm over (batch, d_model, seq_len)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src.permute(1, 2, 0)).permute(2, 0, 1)
        return src


class TSTransformerEncoder(nn.Module):
    """Encoder used for pretraining. Output: (batch, seq_len, feat_dim) reconstruction."""

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward,
                 dropout=0.1, pos_encoding="fixed", activation="gelu", norm="BatchNorm", freeze=False):
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.feat_dim = feat_dim

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == "LayerNorm":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout * (1.0 - freeze), activation=activation
            )
        elif norm == "BatchNorm":
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout * (1.0 - freeze), activation=activation
            )
        else:
            raise ValueError(f"norm should be BatchNorm/LayerNorm, not {norm}")

        # enable_nested_tensor=False keeps padded positions computed the same way in train and eval.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)

        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)

        self.build_output_module()

    def build_output_module(self):
        self.output_layer = nn.Linear(self.d_model, self.feat_dim)

    def embed(self, X, padding_masks):
        """X: (batch, seq_len, feat_dim). padding_masks: (batch, seq_len), True at real steps.
        Returns (batch, seq_len, d_model)."""

        inp = self.project_inp(X.permute(1, 0, 2)) * math.sqrt(self.d_model)  # (seq_len, batch, d_model)
        inp = self.pos_enc(inp)
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)  # the encoder output has no non-linearity
        return self.dropout1(output.permute(1, 0, 2))

    def forward(self, X, padding_masks):
        return self.output_layer(self.embed(X, padding_masks))


class TSTransformerClassifier(TSTransformerEncoder):
    """Encoder plus a classification head. Output: (batch, num_classes) logits, no softmax.

    head:
      "flatten"    mvts TSTransformerEncoderClassiregressor: zero the padding, flatten
                   (batch, max_len * d_model), one Linear. Needs inputs padded to max_len.
      "mean"       masked mean over real time steps, then Linear(d_model, num_classes).
      "attention"  masked softmax attention pooling, then Linear(d_model, num_classes).
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding="fixed", activation="gelu", norm="BatchNorm", freeze=False,
                 head="flatten"):
        self.num_classes = num_classes
        self.head = head
        super().__init__(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward,
                         dropout, pos_encoding, activation, norm, freeze)

    def build_output_module(self):
        if self.head == "flatten":
            self.output_layer = nn.Linear(self.d_model * self.max_len, self.num_classes)
        elif self.head in ("mean", "attention"):
            self.output_layer = nn.Linear(self.d_model, self.num_classes)
            if self.head == "attention":
                self.output_attention = nn.Linear(self.d_model, 1)
        else:
            raise ValueError(f"head should be flatten/mean/attention, not {self.head}")

    def forward(self, X, padding_masks):
        output = self.embed(X, padding_masks) * padding_masks.unsqueeze(-1)  # zero-out padding embeddings

        if self.head == "flatten":
            return self.output_layer(output.reshape(output.shape[0], -1))

        if self.head == "mean":
            pooled = output.sum(dim=1) / padding_masks.sum(dim=1, keepdim=True)
        else:
            scores = self.output_attention(output).squeeze(-1).masked_fill(~padding_masks, float("-inf"))
            pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * output).sum(dim=1)

        return self.output_layer(pooled)
