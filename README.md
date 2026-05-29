# SpaceAx AI — Conversational AI Engine

SpaceAx AI adalah mesin percakapan berbasis **Transformer decoder-only** yang dilatih dari nol (bukan model HuggingFace siap pakai). Fitur utama: memori konteks percakapan, mesin emosi, integrasi **KBBI**, pencarian internet via **ddgs**, augmentasi data anti-hafalan, dan skala model **ProMax** (~1.2B / ~4B / ~8B parameter).

**Pengembang:** Thomas Alfareno Ananta Nugraha — Teknik Informatika, FTEIC, ITS Surabaya  
**Versi:** 2.0.0

---

## Daftar isi

1. [Fitur terbaru](#fitur-terbaru)
2. [Struktur proyek](#struktur-proyek)
3. [Persyaratan sistem](#persyaratan-sistem)
4. [Instalasi](#instalasi)
5. [Cara pengoperasian](#cara-pengoperasian)
6. [ProMax: tier 1B / 4B / 8B](#promax-tier-1b--4b--8b)
7. [Training: tips & ekspektasi](#training-tips--ekspektasi)
8. [Chat & checkpoint](#chat--checkpoint)
9. [Colab / Google Drive](#colab--google-drive)
10. [Pemecahan masalah](#pemecahan-masalah)

---

## Fitur terbaru

| Area | Perilaku |
|------|----------|
| **Chat konteks** | Riwayat STM (`User:` / `AI:`) ikut masuk prompt generasi, bukan hanya pesan terakhir |
| **Follow-up** | Jawaban seperti "kabar ku baik" dikenali setelah AI menanyakan kabar |
| **Augmentasi training** | Paraphrase on-the-fly + variasi intent (`composition_variants`) agar model merangkai kata, bukan hafal satu kalimat |
| **ProMax tier** | `promax_1b`, `promax_4b`, `promax_8b` — auto atau paksa via CLI/env |
| **Generasi adaptif** | Chat mencoba output neural dulu (validator ketat/longgar sesuai `val_loss` checkpoint), lalu fallback |
| **KBBI** | Corpus kamus di tokenizer + pasangan training definisi |
| **Internet** | Pencarian teks Indonesia via paket **`ddgs`** (`learning/internet.py`) |

---

## Struktur proyek

```
SpaceaxAiDebug/
├── main.py              # CLI: train, chat, learn, retrain, test
├── chat.py              # Antarmuka chat interaktif + fallback
├── requirements.txt
├── core/
│   ├── config.py        # Profil model & training
│   ├── promax.py        # Tier 1B/4B/8B
│   ├── model.py         # Transformer SpaceAx
│   ├── tokenizer.py     # BPE
│   └── kbbi.py          # Integrasi KBBI
├── training/
│   ├── dataset.py       # Dataloader + augmentasi dinamis
│   ├── trainer.py       # Training loop, checkpoint, sample epoch
│   ├── text_augment.py
│   ├── composition_variants.py
│   └── generate_seed_data.py
├── learning/
│   ├── internet.py      # Pencarian ddgs
│   └── web_learner.py
├── memory/              # STM/LTM, vector store
├── personality/         # Emosi & preferensi
├── data/
│   ├── seed/            # conversations.json
│   ├── checkpoints/     # model_best.pt, model_epoch_*.pt
│   └── vocab/
└── kbbi/                # Ekstrak kbbi_v_part.zip di sini
```

---

## Persyaratan sistem

- **Python:** 3.10–3.12 disarankan (3.14 bisa dipakai jika PyTorch mendukung)
- **RAM:** auto-detect profil model (lihat tabel ProMax)
- **GPU:** opsional; CUDA mempercepat training (Colab T4, L4, A100, dll.)
- **KBBI:** wajib diekstrak sebelum training penuh

### Dependensi Python (`requirements.txt`)

| Paket | Fungsi |
|-------|--------|
| `torch` | Model & training |
| `ddgs` | **Wajib** — pencarian internet (`from ddgs import DDGS`) |
| `rich` | Tampilan terminal chat |
| `tokenizers` | BPE backend |
| `beautifulsoup4`, `requests` | Web learning |
| `numpy` | Operasi numerik |

> **Penting:** Kode memakai **`ddgs`**, bukan `duckduckgo_search`. Jika muncul `ModuleNotFoundError: No module named 'ddgs'`, jalankan `pip install ddgs>=9.14.0`.

---

## Instalasi

### Windows

1. Instal Python dari [python.org](https://www.python.org/) (centang **Add to PATH**).
2. Clone / unduh repo, masuk ke folder proyek:
   ```cmd
   cd SpaceaxAiDebug
   ```
3. Ekstrak **`kbbi/kbbi_v_part.zip`** ke folder `kbbi/` (isi file `.txt` kamus).
4. Virtual environment (disarankan):
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. Instal dependensi:
   ```cmd
   pip install -U pip
   pip install -r requirements.txt
   pip install ddgs>=9.14.0
   ```
6. (Opsional) PyTorch dengan CUDA — ikuti [pytorch.org](https://pytorch.org/) sesuai driver NVIDIA Anda.

### Linux (Arch, Ubuntu, dll.)

```bash
git clone <url-repo-anda>
cd SpaceaxAiDebug
unzip -o kbbi/kbbi_v_part.zip -d kbbi/   # sesuaikan path zip Anda

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install ddgs>=9.14.0
```

Di Arch, gunakan venv agar tidak bentrok dengan paket sistem (PEP 668).

### Verifikasi instalasi

```bash
python -c "from ddgs import DDGS; import torch; print('ddgs OK, torch', torch.__version__)"
```

Jika gagal pada `ddgs`, instal ulang:

```bash
pip install --upgrade "ddgs>=9.14.0"
```

---

## Cara pengoperasian

Semua perintah lewat `main.py`:

```bash
python main.py <perintah> [opsi]
```

### 1. Training

Melatih model dari seed data + KBBI (+ variasi komposisi):

```bash
python main.py train
```

**Opsi**

| Opsi | Keterangan |
|------|------------|
| `--size` | `small`, `medium`, `large`, `ultra`, `promax` (default: auto dari RAM) |
| `--promax-tier` | `promax_1b`, `promax_4b`, `promax_8b` — paksa sub-tier ProMax |
| `--epochs` | Jumlah epoch (ProMax default minimal ~30 kecuali di-override) |
| `--batch-size` | Batch per device |
| `--grad-accum` | Gradient accumulation (effective batch = batch × accum) |
| `--regen` | Buat ulang seed + tokenizer dari awal |

**Contoh**

```bash
# Colab / GPU menengah — tier 1B
python main.py train --size promax --promax-tier promax_1b

# Paksa 4B (butuh VRAM besar)
python main.py train --size promax --promax-tier promax_4b --batch-size 1 --grad-accum 16

# Eksperimen cepat profil kecil
python main.py train --size medium --epochs 40

# Regenerasi data setelah ubah seed generator
python main.py train --regen
```

Checkpoint terbaik: `data/checkpoints/model_best.pt`  
Checkpoint per epoch: `data/checkpoints/model_epoch_<N>.pt`

### 2. Chat

```bash
python main.py chat
python main.py chat --mode chatdev
python main.py chat --size promax --promax-tier promax_1b
```

Atau langsung:

```bash
python chat.py
```

Chat memakai fallback cerdas jika checkpoint belum matang (`val_loss` tinggi). Setelah training bagus (`val_loss` ≲ 3.5–4), generasi neural diprioritaskan.

### 3. Belajar topik dari internet

```bash
python main.py learn "transformer neural network"
```

Membutuhkan **`ddgs`** terinstal.

### 4. Retrain dari log percakapan

Setelah beberapa sesi `chat`, riwayat disimpan; gabung ke seed lalu latih ulang:

```bash
python main.py retrain
python main.py retrain --size promax --epochs 20 --promax-tier promax_1b
```

### 5. Tes otomatis modul

```bash
python main.py test
# atau
python main.py chatdev
```

---

## ProMax: tier 1B / 4B / 8B

ProMax **bukan** unduh model LLM eksternal — ini arsitektur SpaceAx dengan ukuran berbeda.

| Tier | ~Parameter | Vocab | RAM disarankan | VRAM training (kasar) |
|------|------------|-------|----------------|------------------------|
| `promax_1b` | ~1.2B | 64k | 48 GB | 16 GB (T4, ketat) |
| `promax_4b` | ~4B | 96k | 64 GB | ≥24 GB |
| `promax_8b` | ~8B | 128k | 96 GB | ≥40 GB (A100 40GB+) |

**Pemilihan otomatis** (tanpa paksa): RAM ≥96 & VRAM ≥20 → 8B; RAM ≥64 → 4B; selain itu → 1B.

**Paksa tier**

```bash
export SPACEAX_PROMAX_TIER=promax_4b
python main.py train --size promax
```

atau:

```bash
python main.py train --size promax --promax-tier promax_8b
```

**Colab T4 (16 GB):** gunakan `promax_1b` saja; 4B/8B hampir pasti OOM.

Checkpoint **tidak kompatibel** antar tier (ukuran layer & vocab berbeda).

---

## Training: tips & ekspektasi

- **Batch besar** menstabilkan gradien; **tidak** menggantikan epoch cukup atau membuat loss rendah dalam 1 epoch.
- **Effective batch** = `--batch-size` × `--grad-accum` (ditampilkan saat training mulai).
- **1 epoch** pada ProMax 1B: val_loss biasanya masih tinggi (6–8+), chat belum koheren — normal.
- **Target chat enak:** `val_loss` < ~4 (ideal < 3.5), ProMax sering butuh **≥15–30 epoch**.
- **Low RAM:** optimizer **Adafactor** otomatis untuk model besar; `num_workers=0` di CPU/RAM terbatas.

---

## Chat & checkpoint

| `val_loss` (checkpoint) | Perilaku chat |
|-------------------------|---------------|
| ≤ 3.5 | Generasi neural prioritas utama |
| 3.5 – 5.5 | Model dicoba (validator lebih longgar), lalu fallback |
| 5.5 – 7.5 | Mode awal — dicoba sebentar, fallback dominan |
| > 7.5 | Fallback + memori konteks |

File checkpoint default yang dimuat chat: `data/checkpoints/model_best.pt`.

---

## Colab / Google Drive

Contoh alur di notebook:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependensi (ddgs wajib)
!pip install -q -r requirements.txt
!pip install -q "ddgs>=9.14.0"

# Opsional: paksa tier
import os
os.environ["SPACEAX_PROMAX_TIER"] = "promax_1b"

# Training
!python main.py train --size promax --regen
```

Simpan checkpoint ke Drive jika perlu:

```python
# Salin setelah training
!cp data/checkpoints/model_best.pt "/content/drive/MyDrive/SpaceAX-AI-BETA/data/checkpoints/"
```

Jangan latih ProMax hanya **4 epoch** lalu mengharapkan chat matang — gunakan early stopping default atau `--epochs` tinggi.

---

## Pemecahan masalah

### `ModuleNotFoundError: No module named 'ddgs'`

```bash
pip install "ddgs>=9.14.0"
```

Pastikan venv aktif (`which python` menunjuk ke `.venv`).

### Pencarian internet gagal / kosong

- Cek koneksi jaringan Colab/server.
- Upgrade ddgs: `pip install -U ddgs`
- Beberapa region memblokir DuckDuckGo — coba VPN atau runtime lain.

### CUDA OOM saat training ProMax

- Turunkan tier: `--promax-tier promax_1b`
- `--batch-size 1 --grad-accum 8` (atau lebih besar accum, batch tetap 1)
- Profil lebih kecil: `--size large` atau `medium`

### Chat selalu jawaban template ("belum mendalami…")

- Checkpoint masih mentah (`val_loss` > 7) — lanjutkan training.
- Pastikan `data/checkpoints/model_best.pt` ada dan tier chat sama dengan saat train.

### Tokenizer / checkpoint tidak cocok

- Setelah ubah tier atau vocab, jalankan `python main.py train --regen`.
- Jangan load checkpoint 4B dengan arsitektur 1B.

### KBBI tidak ditemukan

Ekstrak `kbbi/kbbi_v_part.zip`. Tanpa KBBI, training tetap jalan tetapi vocab lebih miskin.

---

## Variabel lingkungan

| Variabel | Fungsi |
|----------|--------|
| `SPACEAX_PROMAX_TIER` | `promax_1b`, `promax_4b`, `promax_8b` |

---

## Tentang pengembang

**Thomas Alfareno Ananta Nugraha**  
Program Studi Teknik Informatika — FTEIC  
Institut Teknologi Sepuluh Nopember (ITS) Surabaya

---

*Dokumentasi ini mencerminkan pipeline SpaceAx AI v2 (ProMax, augmentasi, ddgs, CLI `--promax-tier`). Perbarui README ini jika menambah perintah atau dependensi baru.*
