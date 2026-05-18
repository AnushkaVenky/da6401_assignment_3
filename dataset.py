"""
dataset.py - Multi30k DE->EN dataset wrapper
DA6401 Assignment 3
"""

from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


def _load_spacy_models():
    """Lazy-load spaCy German + English tokenizers."""
    import spacy
    try:
        nlp_de = spacy.load("de_core_news_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("de_core_news_sm")
        nlp_de = spacy.load("de_core_news_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    try:
        nlp_en = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp_en = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    return nlp_de, nlp_en


def _tokenize(nlp, sentence: str) -> List[str]:
    return [t.text.lower() for t in nlp(sentence.strip())]


def build_vocab(
    token_lists: List[List[str]],
    min_freq: int = 2,
    max_size: Optional[int] = None,
) -> Dict[str, int]:
    """Build a {token -> index} dict starting with the four specials."""
    counter: Counter = Counter()
    for toks in token_lists:
        counter.update(toks)

    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

    most_common = counter.most_common(max_size)
    for tok, freq in most_common:
        if freq < min_freq:
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
    return vocab


class VocabWrap:
    """
    Thin wrapper that exposes both ``v["word"]`` (dict-style)
    and ``v.itos[idx]`` / ``v.lookup_token(idx)`` for compatibility
    with the autograder's ``evaluate_bleu`` API.
    """

    def __init__(self, stoi: Dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = {i: t for t, i in stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    def __contains__(self, key) -> bool:
        return key in self.stoi

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.stoi.get(key, self.stoi["<unk>"])
        return self.itos[key]

    def get(self, key: str, default=None):
        return self.stoi.get(key, default)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, "<unk>")


class Multi30kDataset(Dataset):
    """
    Loads Multi30k (bentrevett/multi30k from Hugging Face) and tokenises
    with spaCy.  Vocabularies are built on the training split and reused
    across val / test.
    """

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[VocabWrap] = None,
        tgt_vocab: Optional[VocabWrap] = None,
        min_freq: int = 2,
        max_len: int = 100,
    ):
        from datasets import load_dataset

        self.split = split
        self.max_len = max_len

        # HF download
        hf_split = {"train": "train", "valid": "validation",
                    "validation": "validation", "test": "test"}[split]
        ds = load_dataset("bentrevett/multi30k", split=hf_split)

        nlp_de, nlp_en = _load_spacy_models()
        self.nlp_de = nlp_de
        self.nlp_en = nlp_en

        # Tokenise everything up-front - Multi30k is tiny (~29k rows).
        self.src_tokens: List[List[str]] = []
        self.tgt_tokens: List[List[str]] = []
        for row in ds:
            self.src_tokens.append(_tokenize(nlp_de, row["de"]))
            self.tgt_tokens.append(_tokenize(nlp_en, row["en"]))

        if src_vocab is None:
            stoi = build_vocab(self.src_tokens, min_freq=min_freq)
            src_vocab = VocabWrap(stoi)
        if tgt_vocab is None:
            stoi = build_vocab(self.tgt_tokens, min_freq=min_freq)
            tgt_vocab = VocabWrap(stoi)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Pre-convert to integer tensors.
        self.src_ids: List[List[int]] = [
            self._encode(toks, src_vocab) for toks in self.src_tokens
        ]
        self.tgt_ids: List[List[int]] = [
            self._encode(toks, tgt_vocab) for toks in self.tgt_tokens
        ]

    def _encode(self, toks: List[str], vocab: VocabWrap) -> List[int]:
        toks = toks[: self.max_len - 2]
        return [SOS_IDX] + [vocab.get(t, UNK_IDX) for t in toks] + [EOS_IDX]

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.src_ids[idx], dtype=torch.long),
            torch.tensor(self.tgt_ids[idx], dtype=torch.long),
        )

    # --- Compatibility methods (mentioned in the skeleton) ---

    def build_vocab(self) -> Tuple[VocabWrap, VocabWrap]:
        return self.src_vocab, self.tgt_vocab

    def process_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        return self.src_ids, self.tgt_ids


# ============================================================
# Padding-aware collate function
# ============================================================

def collate_batch(batch, pad_idx: int = PAD_IDX) -> Tuple[torch.Tensor, torch.Tensor]:
    srcs, tgts = zip(*batch)
    src_max = max(s.size(0) for s in srcs)
    tgt_max = max(t.size(0) for t in tgts)

    src_pad = torch.full((len(srcs), src_max), pad_idx, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), tgt_max), pad_idx, dtype=torch.long)
    for i, s in enumerate(srcs):
        src_pad[i, : s.size(0)] = s
    for i, t in enumerate(tgts):
        tgt_pad[i, : t.size(0)] = t
    return src_pad, tgt_pad


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 0,
    min_freq: int = 2,
    max_len: int = 100,
):
    """Convenience wrapper returning train/val/test loaders + vocabs."""
    from torch.utils.data import DataLoader

    train_ds = Multi30kDataset("train", min_freq=min_freq, max_len=max_len)
    val_ds = Multi30kDataset(
        "valid",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        min_freq=min_freq,
        max_len=max_len,
    )
    test_ds = Multi30kDataset(
        "test",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        min_freq=min_freq,
        max_len=max_len,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_batch,
    )
    return train_loader, val_loader, test_loader, train_ds.src_vocab, train_ds.tgt_vocab
