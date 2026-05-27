"""
SpaceaxAI - Data Loader & Dataset
Memproses format data percakapan menjadi input/target tensor untuk model causal LM.

Response-only loss masking:
  Format sequence: [BOS] user_tokens [EMO_*] ai_tokens [EOS]
  Loss hanya dihitung pada ai_tokens dan [EOS].
  Token sebelum dan termasuk [EMO_*] mendapat label -100 (diabaikan CrossEntropyLoss).
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional


class ConversationDataset(Dataset):
    """Dataset untuk causal language modeling dari data percakapan.

    Setiap sample menghasilkan (input_ids, labels) di mana:
      - input_ids: tokens[:-1]   (konteks untuk model)
      - labels   : tokens[1:]    (target prediksi)
    Labels di-mask ke -100 untuk semua posisi sebelum respons AI,
    sehingga CrossEntropyLoss hanya menghitung loss pada token respons.
    """

    # ID token emosi (5-13) — digunakan sebagai pemisah user ↔ AI
    EMO_TOKEN_IDS = set(range(5, 14))

    def __init__(self, data_file: str, tokenizer, max_seq_len: int = 512,
                 augment: bool = True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment

        self.pad_id = tokenizer.special_tokens["<PAD>"]
        self.bos_id = tokenizer.special_tokens["<BOS>"]
        self.eos_id = tokenizer.special_tokens["<EOS>"]

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.conversations = data.get('conversations', [])

        # Setiap elemen: (full_seq, response_start_idx)
        # response_start_idx = indeks pertama token AI dalam full_seq
        self.samples: List[Tuple[List[int], int]] = []
        self._prepare_data()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _resolve_emotion_id(self, emotion_str: str) -> int:
        """Dapatkan ID token emosi, fallback ke EMO_NEUTRAL."""
        emo_token_str = f"<EMO_{emotion_str.upper()}>"
        if emo_token_str in self.tokenizer.special_tokens:
            return self.tokenizer.special_tokens[emo_token_str]
        return self.tokenizer.special_tokens["<EMO_NEUTRAL>"]

    def _augment_text(self, text: str) -> str:
        """Augmentasi ringan: acak upper/lower huruf pertama."""
        if not text:
            return text
        if random.random() < 0.3:
            return text[0].swapcase() + text[1:]
        return text

    def _build_and_truncate(self, user_tokens: List[int], emo_id: int,
                            ai_tokens: List[int]) -> Tuple[List[int], int]:
        """Bangun sequence lengkap dan truncate jika perlu.

        Format: [BOS] user_tokens [EMO] ai_tokens [EOS]
        Prioritas: pertahankan respons AI utuh, potong user input jika perlu.

        Returns:
            (full_seq, response_start)
            response_start = indeks token AI pertama di full_seq
        """
        # Overhead tetap: BOS + EMO + EOS = 3 token
        overhead = 3
        max_content = self.max_seq_len - overhead  # ruang untuk user + ai tokens

        if max_content <= 0:
            # max_seq_len terlalu kecil, minimal sequence
            full_seq = [self.bos_id, emo_id, self.eos_id]
            return full_seq, 2  # response_start menunjuk EOS

        total_content = len(user_tokens) + len(ai_tokens)

        if total_content <= max_content:
            # Muat semua
            pass
        elif len(ai_tokens) <= max_content:
            # Potong user tokens, pertahankan AI utuh
            avail_user = max_content - len(ai_tokens)
            # Ambil bagian akhir user tokens (konteks terdekat lebih penting)
            user_tokens = user_tokens[-avail_user:] if avail_user > 0 else []
        else:
            # AI tokens sendiri sudah melebihi kapasitas — potong AI juga
            # Sisakan minimal 10 token user untuk konteks
            min_user = min(10, len(user_tokens))
            avail_ai = max_content - min_user
            user_tokens = user_tokens[-min_user:]
            ai_tokens = ai_tokens[:max(avail_ai, 1)]

        full_seq = [self.bos_id] + user_tokens + [emo_id] + ai_tokens + [self.eos_id]
        response_start = len(user_tokens) + 2  # +1 BOS, +1 EMO
        return full_seq, response_start

    def _prepare_data(self):
        """Tokenize semua percakapan dan simpan sebagai samples."""
        for conv in self.conversations:
            user_text = conv.get("input", "").strip()
            ai_text = conv.get("response", "").strip()
            emotion = conv.get("emotion", "neutral")

            if not user_text or not ai_text:
                continue

            emo_id = self._resolve_emotion_id(emotion)

            # Encode teks
            user_tokens = self.tokenizer.encode(user_text)
            ai_tokens = self.tokenizer.encode(ai_text)

            full_seq, response_start = self._build_and_truncate(
                user_tokens, emo_id, ai_tokens
            )

            self.samples.append((full_seq, response_start))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        full_seq, response_start = self.samples[idx]

        # Opsional: augmentasi saat training (rebuild jika augment aktif)
        # Karena kita menyimpan token, augmentasi dilakukan di level token-ID
        # (kita hanya bisa augmentasi teks sebelum tokenisasi, tapi itu mahal).
        # Pendekatan efisien: augmentasi sudah diterapkan saat _prepare_data
        # atau bisa diabaikan saat inference.

        # Untuk Causal LM:
        #   Input:  tokens[0 : n-1]
        #   Target: tokens[1 : n]
        seq = list(full_seq)  # copy agar tidak mutate
        input_ids = seq[:-1]
        target_ids = seq[1:]

        # ---- Response-only loss masking ----
        # Dalam full_seq: [BOS] user... [EMO] ai... [EOS]
        # response_start = indeks token AI pertama di full_seq
        #
        # Dalam target_ids (= full_seq[1:]):
        #   posisi i di target_ids memprediksi full_seq[i+1]
        #   Kita ingin loss hanya pada token AI dan EOS.
        #   Token AI pertama di full_seq ada di index response_start.
        #   Di target_ids, itu berada di posisi (response_start - 1).
        #   Semua posisi sebelum itu harus -100.

        mask_end = response_start - 1  # posisi terakhir yang di-mask + 1
        labels = [-100] * mask_end + target_ids[mask_end:]

        # Padding ke max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def create_dataloaders(
    data_file: str,
    tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 512,
    split_ratio: float = 0.9,
    augment: bool = True,
    num_workers: Optional[int] = None,
):
    """Buat Train dan Validation DataLoader.

    Args:
        data_file: Path ke file JSON percakapan.
        tokenizer: Instance BPETokenizer.
        batch_size: Ukuran batch.
        max_seq_len: Panjang maksimum sequence (termasuk special tokens).
        split_ratio: Rasio train/total.
        augment: Aktifkan augmentasi data.
        num_workers: Jumlah worker DataLoader (None = auto-detect).
    """
    import multiprocessing

    dataset = ConversationDataset(data_file, tokenizer, max_seq_len, augment=augment)

    # Split train/val
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        # Data terlalu sedikit — gunakan seluruhnya untuk keduanya
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    # Auto-detect num_workers
    if num_workers is None:
        if torch.cuda.is_available():
            num_workers = min(4, multiprocessing.cpu_count() or 0)
        else:
            num_workers = 0  # Pada CPU, overhead multiprocessing tidak worth it

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader
