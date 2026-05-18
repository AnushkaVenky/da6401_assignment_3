"""
model.py - Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"
This file implements the Transformer architecture from Vaswani et al.:
  scaled_dot_product_attention(Q, K, V, mask) -> (out, weights)
  MultiHeadAttention.forward(q, k, v, mask)   -> Tensor
  PositionalEncoding.forward(x)               -> Tensor
  make_src_mask(src, pad_idx)                 -> BoolTensor
  make_tgt_mask(tgt, pad_idx)                 -> BoolTensor
  Transformer.encode(src, src_mask)           -> Tensor
  Transformer.decode(memory, src_mask, tgt, tgt_mask) -> Tensor
"""

import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown  # weights download 
except ImportError:
    gdown = None

SUBMISSION_GDOWN_ID = "1G8om-oFdllQqcrJlUwXpGlRpEMcV8_KJ"

# Defaults used when the autograder constructs `Transformer()` with no arguments. 
DEFAULT_D_MODEL = 256
DEFAULT_N = 3
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FF = 1024
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_LEN = 150
DEFAULT_PAD_IDX = 1
DEFAULT_CKPT_PATH = "submission_weights.pt"


# ============================================================
# Standalone attention function
# ============================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax(Q . K^T / sqrt(d_k)) . V

    Positions where ``mask`` is True are set to -inf prior to softmax.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


# ============================================================
# Mask helpers
# ============================================================

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for the encoder source.

    Returns BoolTensor shape [batch, 1, 1, src_len];
    True = padding position (to be masked out).
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask for the decoder.

    Returns BoolTensor shape [batch, 1, tgt_len, tgt_len];
    True = masked position (padding OR future token).
    """
    tgt_len = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)          # [B,1,1,T]
    causal = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1,
    )                                                              # [T,T]
    causal = causal.unsqueeze(0).unsqueeze(1)                      # [1,1,T,T]
    return pad_mask | causal                                       # broadcast


# ============================================================
# Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al. 2017, sec. 3.2.2).

    Args:
        d_model:   total model dimensionality (must be divisible by num_heads)
        num_heads: number of attention heads h
        dropout:   dropout probability applied to the attention weights
        scale:     if False, omit the 1/sqrt(d_k) factor (used by the
                   2.2 ablation; default True for the standard model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        scale: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = scale

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights: Optional[torch.Tensor] = None  # last forward pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)

        # [B, L, d_model] -> [B, h, L, d_k]
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        if self.scale:
            scores = scores / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                              # [B,h,Lq,d_k]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017, sec. 3.5).

        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    The table is registered as a *buffer* (not a parameter).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                     # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding (drop-in for the 2.4 ablation)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


# ============================================================
# Feed-forward network
# ============================================================

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x W1 + b1) W2 + b2  (Vaswani et al. 2017, sec. 3.3).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================================
# Encoder / Decoder layers
# (Pre-LayerNorm structure: more stable training, justified in the report.)
# ============================================================

class EncoderLayer(nn.Module):
    """Single encoder block: SelfAttn + FFN with Pre-LayerNorm residuals."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout1(self.self_attn(h, h, h, src_mask))
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x


class DecoderLayer(nn.Module):
    """Single decoder block: masked SelfAttn + CrossAttn + FFN, Pre-LN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout1(self.self_attn(h, h, h, tgt_mask))
        h = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(h, memory, memory, src_mask))
        h = self.norm3(x)
        x = x + self.dropout3(self.ffn(h))
        return x


# ============================================================
# Encoder / Decoder stacks
# ============================================================

def _clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Stack of N EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for lyr in self.layers:
            x = lyr(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for lyr in self.layers:
            x = lyr(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ============================================================
# Full Transformer
# ============================================================

class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer.

    AUTOGRADER USAGE:
        model = Transformer().to(device)
        model.eval()
        english = model.infer(german_sentence)

    With no arguments the constructor:
      1. builds the canonical architecture (DEFAULT_* constants),
      2. downloads the trained weights + vocabs from Google Drive
         (SUBMISSION_GDOWN_ID) into DEFAULT_CKPT_PATH,
      3. overrides the architecture with the checkpoint's saved
         model_config (so we are byte-compatible with the saved weights),
      4. loads weights, source/target vocabs and the spaCy DE tokenizer
         - everything `.infer()` needs.
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = DEFAULT_D_MODEL,
        N: int = DEFAULT_N,
        num_heads: int = DEFAULT_NUM_HEADS,
        d_ff: int = DEFAULT_D_FF,
        dropout: float = DEFAULT_DROPOUT,
        max_len: int = DEFAULT_MAX_LEN,
        pad_idx: int = DEFAULT_PAD_IDX,
        pos_enc: str = "sinusoidal",   # or "learned" (sec. 2.4 ablation)
        attn_scale: bool = True,        # sec. 2.2 ablation
        checkpoint_path: Optional[str] = DEFAULT_CKPT_PATH,
        gdown_id: Optional[str] = SUBMISSION_GDOWN_ID,
    ) -> None:
        super().__init__()

        # --- Step 1: maybe download the checkpoint and read its config ---
        ckpt = self._fetch_checkpoint(checkpoint_path, gdown_id)

        if ckpt is not None and "model_config" in ckpt:
            cfg = ckpt["model_config"]
            src_vocab_size = cfg["src_vocab_size"]
            tgt_vocab_size = cfg["tgt_vocab_size"]
            d_model = cfg["d_model"]
            N = cfg["N"]
            num_heads = cfg["num_heads"]
            d_ff = cfg["d_ff"]
            dropout = cfg["dropout"]
            pad_idx = cfg["pad_idx"]

        # If we still have no vocab sizes (no checkpoint AND caller didn't
        # supply any) we can't build the embeddings - bail clearly.
        if src_vocab_size is None or tgt_vocab_size is None:
            raise RuntimeError(
                "Transformer(): no checkpoint loaded and src/tgt vocab "
                "sizes not provided. To use this class with the "
                "autograder, upload your trained weights to Drive and "
                "set SUBMISSION_GDOWN_ID in model.py."
            )

        # --- Step 2: build the architecture ---
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        if pos_enc == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, dropout, max_len)
        else:
            self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        if not attn_scale:
            for m in (enc_layer.self_attn,
                      dec_layer.self_attn, dec_layer.cross_attn):
                m.scale = False

        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not attn_scale:
            for m in self.modules():
                if isinstance(m, MultiHeadAttention):
                    m.scale = False

        # --- Step 3: load weights + vocabs from the checkpoint ---
        self.src_vocab: Optional[dict] = None
        self.tgt_vocab: Optional[dict] = None
        self.src_tokenizer = None

        if ckpt is not None:
            sd = ckpt.get("model_state_dict", ckpt)
            self.load_state_dict(sd, strict=False)
            self.src_vocab = self._coerce_vocab(ckpt.get("src_vocab"))
            self.tgt_vocab = self._coerce_vocab(ckpt.get("tgt_vocab"))

        # --- Step 4: load spaCy DE tokenizer once ---
        if self.src_vocab is not None:
            self.src_tokenizer = self._load_spacy_de()

    # ---- Internal helpers -------------------------------------------

    @staticmethod
    def _fetch_checkpoint(path: Optional[str], gdown_id: Optional[str]):
        """Download (if missing) and load the checkpoint dict."""
        if path is None:
            return None

        if not os.path.exists(path):
            if gdown is None:
                return None
            if gdown_id is None or gdown_id.startswith("<"):
                # Placeholder ID - nothing to download from.
                return None
            print(f"[Transformer] downloading weights from Drive id={gdown_id}")
            gdown.download(id=gdown_id, output=path, quiet=False)

        if not os.path.exists(path):
            return None
        return torch.load(path, map_location="cpu")

    @staticmethod
    def _coerce_vocab(v) -> Optional[dict]:
        """Accept either a dict[str,int] or a VocabWrap; return a plain dict."""
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if hasattr(v, "stoi"):
            return dict(v.stoi)
        return dict(v)

    @staticmethod
    def _load_spacy_de():
        import spacy
        try:
            return spacy.load(
                "de_core_news_sm",
                disable=["tagger", "parser", "ner", "lemmatizer"],
            )
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("de_core_news_sm")
            return spacy.load(
                "de_core_news_sm",
                disable=["tagger", "parser", "ner", "lemmatizer"],
            )

    # ---- Autograder hooks ----

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        hidden = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(hidden)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ---- Single-sentence inference ----

    @torch.no_grad()
    def infer(self, src_sentence: str) -> str:
        """
        End-to-end translation of a single German sentence to English.

        Tokenises with spaCy, encodes to source-vocab IDs, runs greedy
        autoregressive decoding, then detokenises target IDs to a string.
        All state is loaded in `__init__` from the gdown checkpoint - no
        external setup is required.
        """
        assert self.src_vocab is not None and self.tgt_vocab is not None, (
            "Transformer.infer(): vocabs not loaded. The constructor "
            "should have pulled them from the gdown checkpoint."
        )
        assert self.src_tokenizer is not None, (
            "Transformer.infer(): spaCy German tokenizer not loaded."
        )

        device = next(self.parameters()).device
        self.eval()

        sos = self.src_vocab["<sos>"]
        eos = self.src_vocab["<eos>"]
        unk = self.src_vocab["<unk>"]

        tokens = [t.text.lower() for t in self.src_tokenizer(src_sentence)]
        ids = [sos] + [self.src_vocab.get(t, unk) for t in tokens] + [eos]
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, self.pad_idx)

        tgt_sos = self.tgt_vocab["<sos>"]
        tgt_eos = self.tgt_vocab["<eos>"]

        # Autoregressive greedy decoding (inlined to avoid a circular
        # import on train.py).
        memory = self.encode(src, src_mask)
        max_len = min(2 * len(ids) + 10, 100)
        ys = torch.full((1, 1), tgt_sos, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, self.pad_idx)
            logits = self.decode(memory, src_mask, ys, tgt_mask)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            if nxt.item() == tgt_eos:
                break

        itos = {i: w for w, i in self.tgt_vocab.items()}
        words = []
        for idx in ys[0].tolist()[1:]:                       # skip <sos>
            if idx == tgt_eos:
                break
            tok = itos.get(idx, "<unk>")
            if tok in ("<pad>", "<sos>"):
                continue
            words.append(tok)
        return " ".join(words)
