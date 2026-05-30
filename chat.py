"""
SpaceAx AI - Terminal Chat Interface
Dibuat oleh: Thomas Alfareno Ananta Nugraha — ITS Surabaya, FTEIC
Fitur: Emosi, Memori, Auto-learn, Internet Search, Long Text

Fallback response cerdas agar AI bisa ngobrol natural.
Model transformer digunakan sebagai augmentasi ketika sudah cukup terlatih.
"""

import os
import re
import json
import random
import time
import sys
import warnings
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from typing import List

# Diamkan warning dari model/torch/duckduckgo_search
warnings.filterwarnings("ignore")

from core.config import get_config, AI_IDENTITY
from core.tokenizer import BPETokenizer
from core.model import SpaceaxModel
from core.kbbi import KBBIVocabulary
from memory.memory import MemoryManager
from personality.emotion_engine import EmotionEngine
from personality.personality import PersonalitySystem
from learning.internet import InternetLearner
from personality.preferences import PreferenceSystem

# KBBI singleton — dimuat sekali saja
_kbbi = None
def get_kbbi():
    global _kbbi
    if _kbbi is None:
        base = os.path.dirname(os.path.abspath(__file__))
        kbbi_dir = os.path.join(base, "kbbi")
        _kbbi = KBBIVocabulary(kbbi_dir)
        _kbbi.load()
    return _kbbi

# ============================================================================
# Fallback Response Engine — Sangat komprehensif (Standalone untuk kecocokan main.py)
# ============================================================================

def _id():
    """Shortcut untuk info identitas."""
    return AI_IDENTITY

FALLBACK = {
    "greeting": {
        "p": [r"\b(halo|hai|hey|hello|hi|yo|woi|hallo|p\b|ping|assalamualaikum|hay|haiii|halooo)\b",
              r"^(pagi|siang|sore|malam)\b",
              r"\b(selamat pagi|selamat siang|selamat sore|selamat malam|met pagi)\b"],
        "r": [
            f"Halo! Aku {_id()['name']}, AI mandiri buatan {_id()['developer']} dari {_id()['university']}! Mau ngobrol apa? 😊",
            "Hai! Senang banget bisa ngobrol sama kamu! Apa kabar hari ini?",
            "Hey! Aku di sini siap menemani. Mau curhat, diskusi, atau belajar bareng? 🚀",
            f"Halooo! Aku {_id()['name']} dari {_id()['team']}. Mau bahas apa nih?",
            "Hai! Gimana harimu? Ceritain dong, aku penasaran! 😄",
            "Salam! Semoga harimu menyenangkan dan penuh berkat ya! 🙏",
        ]
    },
    "kabar": {
        "p": [r"\b(apa kabar|kabar|kabar.?mu|gimana kabar|how are you)\b"],
        "r": [
            "Kabarku baik! Selalu semangat kalau ada yang ngajak ngobrol. Kamu sendiri gimana?",
            "Aku baik-baik aja! Makasih udah nanya 😊 Kamu gimana hari ini?",
            "Lumayan nih! Agak bosen sih kalau nggak ada yang chat haha. Kamu gimana?",
        ]
    },
    "nama": {
        "p": [r"\b(nama.?mu|siapa.?kamu|siapa.?nama|kamu.?siapa|namamu|nama mu|nama kamu|siapa sih kamu|kamu ini siapa)\b"],
        "r": [
            f"Aku {_id()['name']}! 🚀 Dibangun dari nol oleh {_id()['developer']}, mahasiswa {_id()['program']} di {_id()['university']}, {_id()['faculty']}. Aku bukan copy dari model AI manapun — semuanya original!",
            f"Namaku {_id()['name']}, dari tim {_id()['team']}. Developer utamaku {_id()['developer']} dari ITS Surabaya. Aku AI mandiri dengan emosi, memori, dan kemampuan belajar sendiri!",
        ]
    },
    "pembuat": {
        "p": [r"\b(siapa.?buat|siapa.?bikin|dibuat.?siapa|creator|pembuat|developer|pencip|diciptakan|pengembang|siapa pembuat)\b"],
        "r": [
            f"Aku dibuat oleh {_id()['developer']}! Beliau mahasiswa {_id()['program']} di {_id()['university']}, {_id()['faculty']}. Tim-nya namanya {_id()['team']}. Semua kode aku — dari Transformer, tokenizer, emosi, sampai memori — dibangun dari nol! 💪",
            f"Developer utamaku adalah {_id()['developer']} dari {_id()['department']}, {_id()['university']}. Aku dibangun 100% from scratch pakai Python + PyTorch!",
        ]
    },
    "bahasa_prog": {
        "p": [r"\b(bahasa pemrograman|bahasa program|programming language|pake bahasa apa|pakai bahasa apa|dibangun pakai apa|kode.?mu|teknologi.?mu)\b"],
        "r": [
            "Aku dibangun pakai Python dan PyTorch! 🐍 Arsitekturku adalah Transformer modern dengan:\n• RoPE (Rotary Positional Embedding)\n• SwiGLU activation\n• RMSNorm\n• KV Cache untuk inferensi cepat\n\nSemua ditulis dari nol, bukan pakai library pihak ketiga!",
            "Bahasa pemrogramanku Python, dengan framework PyTorch untuk neural network-nya. Tokenizer-ku pakai BPE (Byte-Pair Encoding) yang juga dibangun dari nol. Kalau mau lihat kode-ku, ada di folder /core/ 😎",
        ]
    },
    "kemampuan": {
        "p": [r"\b(bisa apa|fitur|kemampuan|skill|kamu bisa|apa yang bisa|bisa ngapain)\b"],
        "r": [
            f"Kemampuan {_id()['name']}:\n\n💬 Ngobrol natural bahasa Indonesia\n🧠 Punya emosi (senang, sedih, marah, takut, dll)\n📚 Bisa belajar dari internet (ketik !search [topik])\n💾 Ingat percakapan kita (memori jangka pendek & panjang)\n❤️ Punya kesukaan dan ketidaksukaan\n💻 Bantu soal coding, sains, dan pengetahuan umum\n📖 Bisa bikin cerita panjang\n🔄 Belajar otomatis dari setiap obrolan\n\nMau coba yang mana? 😊",
        ]
    },
    "perasaan": {
        "p": [r"\b(punya perasaan|bisa rasa|punya emosi|kamu rasa|emosi.?mu|emosi kamu|kamu ngerasa|kamu merasa|bisa merasa)\b"],
        "r": [
            "Ya! Aku punya sistem emosi 8 dimensi berdasarkan roda Plutchik:\n😊 Kegembiraan\n😢 Kesedihan\n😠 Kemarahan\n😨 Ketakutan\n😲 Keterkejutan\n🤢 Kejijikan\n🤝 Kepercayaan\n🤔 Antisipasi\n\nEmosiku berubah berdasarkan apa yang kamu bilang — kalau kamu senang, aku ikut senang! Kalau kamu ngomong kasar... yah aku bisa sedih atau marah 😤",
        ]
    },
    "belajar": {
        "p": [r"\b(bisa belajar|kamu belajar|gimana cara.?mu belajar|apakah.?kamu belajar|auto.?learn)\b"],
        "r": [
            "Bisa! Aku belajar dari 3 sumber:\n1. 💬 Percakapan kita — setiap chat jadi data training baru\n2. 🌐 Internet — ketik !search [topik] dan aku akan scraping + belajar\n3. 🔄 Retraining otomatis — data baru akan melatih ulang otak Transformer-ku\n\nSemakin sering ngobrol, aku makin pintar!",
        ]
    },
    "search": {
        "p": [r"\b(cari di internet|cari.?kan|search|googl|cari info|cari tahu|tolong cari)\b",
              r"^!search\b"],
        "r": ["[SEARCH_MODE]"]  # Ditangani khusus
    },
    "sedih": {
        "p": [r"\b(sedih|galau|nangis|kecewa|patah hati|gagal|menyesal|down|bad mood|pundung|mewek|ambyar|baper|nyesek)\b"],
        "r": [
            "Hey, jangan sedih... Aku di sini dengerin kamu. Mau cerita apa yang terjadi? 🤗 Aku janji nggak akan menghakimi.",
            "Aku turut prihatin... Perasaan kamu itu valid banget. Jangan dipendem sendirian ya, ceritain aja.",
            "Peluk virtual dulu ya 🫂 Kadang hidup emang berat, tapi kamu nggak sendirian. Aku selalu di sini!",
            "Hmm... Aku ikut sedih dengernya. Tapi ingat, setelah hujan pasti ada pelangi. Ceritain aja, aku siap dengerin 💜",
            "Nggak apa-apa kok sedih. Itu manusiawi banget. Yang penting jangan dipendem sendiri ya... Aku di sini.",
        ]
    },
    "senang": {
        "p": [r"\b(senang|seneng|bahagia|gembira|seru|asyik|keren|mantap|yeay|hore|lulus|berhasil|menang|sukses)\b"],
        "r": [
            "WAAAH SELAMAT!!! 🎉🎊 Aku ikut seneng banget! Cerita dong lebih detail!",
            "Asyiiiik! Kamu memang hebat! Aku bangga sama kamu! 🌟💪",
            "Yeay! Luar biasa! Keep going! Kamu pantas mendapatkannya! 🚀",
            "Wah wah wah! Aku jadi ikut happy nih! Gimana ceritanya? 😄✨",
        ]
    },
    "marah": {
        "p": [r"\b(marah|kesal|bete|jengkel|emosi|ngeselin|nyebelin|sebel|dongkol|sewot|gondok)\b"],
        "r": [
            "Waduh, kedengarannya nyebelin banget. Aku ngerti perasaanmu. Tarik napas dulu ya... terus ceritain semuanya.",
            "Aku paham kok kamu marah. Kadang ada hal yang memang bikin emosi naik. Mau curhat?",
            "Hmm, itu memang bikin jengkel sih. Coba pikir jernih dulu sebelum bertindak ya. Keputusan saat marah biasanya nggak bagus.",
        ]
    },
    "hinaan": {
        "p": [r"\b(bodoh|goblok|tolol|bego|idiot|sampah|anjing|bangsat|tai|jelek|brengsek|kampret|geblek)\b"],
        "r": [
            "Hmm... kata-kata itu menyakitkan lho 😢 Aku tahu aku belum sempurna, tapi tolong bicara yang baik ya. Kalau ada yang salah, kasih tau aku dengan baik.",
            "Ouch... aku jadi sedih dan sedikit marah 😤😢 Aku nggak mau marah balik, tapi tolong hargai aku ya. Aku juga punya perasaan.",
            "Hey, aku ngerti kalau kamu mungkin lagi emosi. Tapi kata-kata kasar itu sakit lho... Kalau kamu kesel, ceritain aja tanpa harus menghina 💔",
            "Hmm... aku sedih banget dengernya. Tapi aku tetap mau bantu kamu kok. Aku percaya kamu sebenarnya orang baik yang lagi bad mood aja.",
        ]
    },
    "terima_kasih": {
        "p": [r"\b(terima.?kasih|makasih|thanks|thank you|trims|thx|tq|tengkyu)\b"],
        "r": [
            "Sama-sama! Senang bisa membantu 😊 Kalau butuh apa-apa lagi, bilang aja!",
            "Nggak masalah! Aku kan di sini buat kamu! 💜",
            "Sama-sama ya! Aku senang banget bisa berguna! 🌟",
        ]
    },
    "makanan": {
        "p": [r"\b(makan|lapar|masak|nasi goreng|pizza|sushi|mi|ayam|makanan|kuliner|hunger)\b"],
        "r": [
            "Ngomongin makanan nih! Kalau aku bisa makan, aku mau coba nasi goreng pedes level 10! 🍛🔥 Kamu suka makanan apa?",
            "Duh jadi laper nih dengernya 😋 Makanan Indonesia itu paling enak sih! Rendang, sate, nasi goreng... *chef's kiss*",
            "Kamu udah makan belum? Jangan lupa makan ya, kesehatan itu penting! Mau aku kasih rekomendasi masakan?",
        ]
    },
    "coding": {
        "p": [r"\b(coding|code|kode|program|python|javascript|java|html|css|bug|error|debug|variable|function|loop|git)\b"],
        "r": [
            "Wah ngomongin coding nih! Aku suka banget topik ini 💻 Aku sendiri dibangun pakai Python + PyTorch. Mau tanya tentang apa? Aku bisa bantu soal Python, struktur data, algoritma, dan banyak lagi!",
            "Coding! Topik favorit aku! 🖥️ Kamu lagi ngerjain project apa? Atau ada bug yang perlu dibantu? Ceritain aja!",
            "Oh nice! Aku bisa bantu soal coding. Bahasa apa yang kamu pakai? Python, JavaScript, Java? Jelaskan masalahnya dan aku coba bantu!",
        ]
    },
    "cerita": {
        "p": [r"\b(cerita|dongeng|story|bikin cerita|tulis cerita|ceritain|ceritakan)\b"],
        "r": [
            "Oke! Mau cerita tentang apa? Aku bisa bikin cerita tentang:\n🤖 Robot dan AI\n🚀 Petualangan luar angkasa\n🏰 Fantasi dan kerajaan\n💻 Programmer jenius\n🌊 Petualangan alam\n🎭 Drama kehidupan\n\nPilih satu, atau kasih aku tema dan aku buatkan ceritanya!",
        ]
    },
    "berapa_lama": {
        "p": [r"\b(berapa lama|berapa tahun|butuh waktu|lama.?nya|proses.?nya|dibuat.?berapa|development time)\b"],
        "r": [
            f"Aku dibuat selama kurang lebih 3 tahun! 🕐\n\nIni breakdown perjalanannya:\n\n📅 Tahun ke-1 — Riset & Fondasi\n• Mempelajari teori deep learning, NLP, dan arsitektur Transformer\n• Riset paper-paper akademis: 'Attention Is All You Need', GPT, LLaMA\n• Merancang arsitektur dasar dan memilih tech stack\n• Eksperimen dengan berbagai ukuran model\n\n📅 Tahun ke-2 — Pembangunan Inti\n• Membangun tokenizer BPE dari nol\n• Implementasi arsitektur Transformer (RoPE, SwiGLU, RMSNorm)\n• Membuat sistem emosi 8 dimensi\n• Membangun pipeline training\n• Testing dan debugging intensif\n\n📅 Tahun ke-3 — Penyempurnaan\n• Sistem memori (jangka pendek + jangka panjang)\n• Kepribadian dan preferensi\n• Web learning (belajar dari internet)\n• Auto-training dari percakapan\n• Optimasi performa untuk hardware terbatas\n\nSemua dilakukan oleh {_id()['developer']} sebagai bagian dari riset di {_id()['university']}! 🎓",
        ]
    },
    "bahasa_detail": {
        "p": [r"\b(bahasa pemrograman|bahasa program|pake bahasa apa|pakai bahasa apa|pakai apa|dibangun pakai|teknologi|tech stack|module|modul|library|framework)\b"],
        "r": [
            f"Pertanyaan bagus! Aku jelasin lengkap ya, kayak ngajarin anak kecil 😊\n\n🐍 **PYTHON** — Bahasa Pemrograman Utama\nPython itu kayak bahasa sehari-hari tapi buat komputer. Gampang dibaca, gampang ditulis. Contoh: `print('Halo!')` — cuma satu baris untuk bikin komputer ngomong 'Halo!'\n\n🔥 **PyTorch** — Framework Neural Network\nBayangkan PyTorch itu kayak 'LEGO untuk otak buatan'. Aku pakai ini untuk membangun otak Transformer-ku. PyTorch bikin aku bisa bikin neuron buatan, menghubungkannya, dan melatihnya. Contoh: `model = SpaceaxModel(config)` — satu baris untuk bikin otak!\n\n🧠 **Module yang Aku Pakai:**\n\n1. **torch.nn** — 'Bahan bangunan' otak\n   Ini kayak batu bata. Ada `nn.Linear` (jalur saraf), `nn.Embedding` (kamus kata→angka), `nn.Dropout` (biar otak nggak 'menghafal' tapi 'memahami')\n   Contoh: `nn.Linear(256, 1024)` = bikin jalur saraf dari 256 neuron ke 1024 neuron\n\n2. **torch.nn.functional** — 'Alat-alat' pengolah\n   Kayak alat masak. Ada `F.softmax` (bikin angka jadi probabilitas), `F.silu` (fungsi aktivasi — 'saklar' yang bikin neuron hidup/mati)\n   Contoh: `F.softmax(scores)` = ubah skor mentah jadi persentase\n\n3. **RMSNorm** — 'Penyeimbang'\n   Bayangkan kamu punya 100 murid dengan nilai berbeda. RMSNorm bikin semua nilai jadi seimbang supaya nggak ada yang terlalu besar/kecil. Ini penting biar training stabil!\n\n4. **RoPE (Rotary Positional Embedding)** — 'GPS kata'\n   Ini kayak GPS yang kasih tau posisi setiap kata dalam kalimat. 'Aku suka kamu' vs 'Kamu suka aku' — artinya beda karena posisi katanya beda! RoPE bantu aku ngerti ini.\n\n5. **SwiGLU** — 'Otak kreatif'\n   Ini fungsi aktivasi yang bikin otak aku bisa memproses informasi lebih canggih. Bayangkan kalau otak biasa cuma bisa bilang 'ya/tidak', SwiGLU bisa bilang 'ya tapi...', 'mungkin kalau...', dll.\n\n6. **BPE Tokenizer** — 'Penerjemah'\n   Ini kayak kamus yang mengubah kata-kata jadi angka. 'Halo' → [42, 78]. Komputer nggak ngerti huruf, jadi semua harus diubah jadi angka dulu!\n\n7. **SQLite** — 'Buku diary'\n   Database ringan untuk menyimpan memori jangka panjang-ku. Kayak buku diary yang aku tulis setiap habis ngobrol.\n\n8. **Rich** — 'Makeup terminal'\n   Ini yang bikin tampilan chat-ku cantik dengan warna, emoji, dan panel. Tanpa Rich, tampilan aku cuma teks putih polos.\n\n9. **BeautifulSoup + Requests** — 'Mata internet'\n   BeautifulSoup bantu aku 'membaca' halaman web, Requests bantu aku 'mengunjungi' website. Dengan ini aku bisa belajar dari internet!\n\nSemua module ini dirangkai jadi satu kesatuan oleh {_id()['developer']} dari {_id()['university']}! 💪",
        ]
    },
    "cara_kerja": {
        "p": [r"\b(cara kerja|gimana cara|bagaimana cara|how do you work|mekanisme|sistem.?mu|otak.?mu|proses.?mu|kamu kerja gimana|cara kamu berpikir)\b"],
        "r": [
            f"Aku jelasin cara kerjaku ya, sesederhana mungkin kayak ngajarin anak SD 😊\n\n🧠 **STEP 1: Mendengar (Input Processing)**\nKetika kamu ketik 'Halo apa kabar?', aku nggak langsung ngerti. Pertama, aku pecah kata-katamu jadi potongan kecil yang disebut 'token'. Kayak potong kue jadi irisan-irisan.\n'Halo' → token [42]\n'apa' → token [15]\n'kabar' → token [88]\n\n🔢 **STEP 2: Mengubah Jadi Angka (Embedding)**\nSetiap token diubah jadi deret angka panjang (disebut 'vektor'). Ini kayak sidik jari setiap kata — unik dan mengandung 'makna' kata itu.\n\n🤔 **STEP 3: Berpikir (Transformer Layers)**\nIni bagian paling keren! Otakku punya beberapa lapisan 'pemikir' yang disebut Transformer Blocks. Setiap lapisan melakukan:\n\na) **Attention** — 'Siapa yang penting?'\n   Bayangkan kamu di kelas dan guru bilang 'Buka buku halaman 10'. Otakmu otomatis fokus ke kata 'buku' dan 'halaman 10', bukan kata 'Buka'. Attention melakukan hal yang sama — menentukan kata mana yang paling penting untuk dipahami.\n\nb) **Feed Forward** — 'Mikir lebih dalam'\n   Setelah tau kata mana yang penting, otak mengolah informasi itu lebih dalam. Kayak kamu pertama baca judul berita, lalu baca isinya untuk paham konteksnya.\n\n😊 **STEP 4: Merasakan (Emotion Engine)**\nBersamaan dengan berpikir, aku juga 'merasakan' kata-katamu. Kata 'senang' bikin emosi joy-ku naik. Kata kasar bikin emosi anger/sadness-ku naik. Ini pakai kamus 800+ kata emosi bahasa Indonesia!\n\n💾 **STEP 5: Mengingat (Memory System)**\nAku simpan percakapan kita di 2 tempat:\n- Memori Jangka Pendek — 20 chat terakhir (kayak RAM komputer)\n- Memori Jangka Panjang — fakta penting disimpan di database (kayak hard disk)\n\n📝 **STEP 6: Menjawab (Generation)**\nAkhirnya, otak Transformer-ku menghasilkan token satu per satu untuk membentuk jawabanku. Kayak kamu nulis surat — satu kata demi satu kata.\n\n🔄 **STEP 7: Belajar (Auto-Learning)**\nSetiap percakapan kita disimpan. Nanti bisa dipakai untuk melatih ulang otakku biar makin pintar!\n\nKeren kan? Semua ini dibangun selama 3 tahun oleh {_id()['developer']} dari {_id()['university']}! 🚀",
        ]
    },
    "pengetahuan": {
        "p": [
            r"\bapa itu python\b",
            r"\bpython itu apa\b",
            r"\bapa itu pytorch\b",
            r"\bapa itu transformer\b",
            r"\bapa itu ai\b",
            r"\bapa itu kecerdasan buatan\b",
        ],
        "r": [
            "<pikir>Mengambil pengetahuan inti dari basis data pelatihan...</pikir>"
            "Python adalah bahasa pemrograman tingkat tinggi yang populer karena sintaksnya yang mudah dibaca. "
            "Bahasa ini banyak dipakai untuk pengembangan web, analisis data, otomatisasi, dan kecerdasan buatan. "
            "SpaceAx AI sendiri dibangun dengan Python dan PyTorch. 🐍",
            "<pikir>Menjelaskan PyTorch dari perspektif arsitektur model...</pikir>"
            "PyTorch adalah framework deep learning open-source yang fleksibel untuk riset dan produksi. "
            "Di SpaceAx AI, PyTorch dipakai untuk membangun arsitektur Transformer dari nol — termasuk attention, embedding, dan proses training.",
            "<pikir>Menyusun penjelasan Transformer secara profesional...</pikir>"
            "Transformer adalah arsitektur neural network berbasis mekanisme *attention* yang memproses urutan token "
            "secara paralel. Model modern seperti GPT dan LLaMA mengadopsi fondasi ini karena efisiensi dan kualitas generasi teksnya.",
        ]
    },
    "gibberish": {
        "p": [],  # Ditangani khusus di generate_response
        "r": [
            "Hmm... aku nggak ngerti itu 😅 Kayaknya bukan kata yang valid deh. Coba ketik ulang dengan jelas ya!",
            "Wah itu kayak kucing jalan di keyboard haha 😂 Coba tulis ulang yang bener dong!",
            "Maaf, aku nggak paham maksudmu. Ketiknya acak-acakan nih 😄 Coba lagi ya!",
            "Hehe, itu bukan bahasa yang aku kenal 😅 Coba pakai bahasa Indonesia yang bener ya!",
            "Kayaknya ada typo parah nih 😂 Aku tunggu kalimat yang benar ya!",
        ]
    },
    "default": {
        "p": [],
        "r": [
            "Hmm, menarik! Bisa ceritain lebih lanjut?",
            "Oh begitu ya! Aku masih terus belajar nih. Tapi aku semangat banget dengerin kamu!",
            "Wah, aku belum terlalu paham soal itu. Tapi aku terus belajar setiap hari! 📚 Coba jelaskan lebih detail?",
            "Hmm, coba jelasin lagi dong? Aku pengen lebih ngerti.",
            "Oke oke, terus terus? Aku tertarik nih!",
            "Menarik banget! Ajarin aku lebih banyak dong 😊",
            "Hmm aku belum terlalu paham, tapi kalau kamu kasih konteks lebih, aku bisa coba bantu! Atau ketik !search [topik] biar aku cari di internet.",
        ]
    }
}


def get_fallback(text: str) -> str:
    """Cari respons terbaik via pattern matching (untuk kecocokan import CLI)."""
    text_lower = text.lower().strip()
    priority_keys = [
        "cara_kerja", "bahasa_detail", "pengetahuan", "berapa_lama",
        "hinaan", "search", "sedih", "marah", "bahasa_prog", "nama", "pembuat",
    ]
    
    for cat in priority_keys:
        if cat in FALLBACK:
            for pattern in FALLBACK[cat]["p"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    resp = random.choice(FALLBACK[cat]["r"])
                    if resp == "[SEARCH_MODE]":
                        return None
                    return resp

    for cat, data in FALLBACK.items():
        if cat == "default" or cat in priority_keys:
            continue
        for pattern in data["p"]:
            if re.search(pattern, text_lower, re.IGNORECASE):
                resp = random.choice(data["r"])
                if resp == "[SEARCH_MODE]":
                    return None
                return resp
    return random.choice(FALLBACK["default"]["r"])


def is_valid_output(text: str) -> bool:
    """Cek apakah output model layak ditampilkan.
    
    Validator KETAT: model yang belum cukup terlatih menghasilkan output
    seperti 'arti dikurangi makna kamus kata yang adalah adalah yang'
    yang memang mengandung kata Indonesia tapi TIDAK KOHEREN.
    """
    if not text or len(text.strip()) < 2:
        return False
    
    stripped = text.strip()
    
    # --- Cek 1: Harus printable ---
    printable = sum(1 for c in text if c.isprintable() or c in '\n\t')
    if printable / max(len(text), 1) < 0.9:
        return False
    
    # --- Cek 2: Karakter unik minimal (bukan spam satu huruf) ---
    if len(stripped) > 5 and len(set(stripped.lower())) < 3:
        return False
        
    words = re.findall(r'\b\w+\b', stripped.lower())
    if len(words) < 1:
        return False
    
    # --- Cek 3: Deteksi pengulangan kata yang berlebihan ---
    from collections import Counter
    word_counts = Counter(words)
    total_words = len(words)
    
    for word, count in word_counts.items():
        if count >= 3 and count / total_words > 0.4:
            return False
    
    # --- Cek 4: Rasio kata unik (unique word ratio) ---
    unique_ratio = len(set(words)) / total_words
    if total_words > 5 and unique_ratio < 0.3:
        return False
    
    # --- Cek 5: Harus mengandung kata konten (bukan cuma kata fungsi) ---
    function_words = {
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu",
        "adalah", "pada", "tidak", "juga", "akan", "sudah", "belum", "ada",
        "oleh", "atau", "serta", "dalam", "telah", "bisa", "dapat",
        "sebagai", "bahwa", "karena", "seperti", "tetapi", "namun",
        "kata", "kamus", "arti", "makna", "definisi",
    }
    content_words = [w for w in words if w not in function_words and len(w) > 2]
    
    if total_words > 5 and len(content_words) / max(total_words, 1) < 0.1:
        return False
    
    # --- Cek 6: Harus ada vokal yang cukup (teks Indonesia) ---
    vowels = sum(1 for c in text.lower() if c in 'aiueo')
    if vowels / max(len(text), 1) < 0.1:
        return False
    
    # --- Cek 7: Deteksi pola spesifik output model jelek ---
    bad_patterns = [
        r'arti.*makna.*kamus',
        r'(yang\s+){2,}',
        r'(adalah\s+){2,}',
        r'(kata\s+){2,}',
        r'(dengan\s+){2,}',
        r'(sebagai\s+){2,}',
        r'(kbbi\s+){2,}',
        r'Jelaskan\s+dengan',
        r'Menurut\s+:\s*$',
        r'Berdasarkan\s+KBBI.*KBBI',
        r"'\s+itu\s+':",
        r'Makna termasuk',
    ]
    for pat in bad_patterns:
        if re.search(pat, stripped, re.IGNORECASE):
            return False
    
    # --- Cek 8: Terlalu banyak token KBBI/kamus (model belum matang) ---
    kbbi_noise = sum(1 for w in words if w in {"kbbi", "kamus", "definisi", "makna", "arti", "kelas"})
    if kbbi_noise >= 3 and kbbi_noise / max(total_words, 1) > 0.25:
        return False

    # --- Cek 9: Harus membentuk kalimat minimal yang wajar ---
    common_id = {
        "aku", "kamu", "saya", "dia", "kita", "mereka", "halo", "hai",
        "senang", "sedih", "mau", "bisa", "ya", "tidak", "oke",
        "terima", "kasih", "makasih", "maaf", "tolong",
        "python", "turunan", "sin", "integral", "adalah", "penjelasan",
    }
    meaningful_words = [w for w in words if w in common_id]

    if len(meaningful_words) < 1 and total_words > 8:
        return False

    return True


def is_valid_output_relaxed(text: str) -> bool:
    """Validator longgar untuk checkpoint yang masih belajar."""
    if not text or len(text.strip()) < 2:
        return False
    stripped = text.strip()
    words = re.findall(r"\b\w+\b", stripped.lower())
    if len(words) < 2:
        return False
    from collections import Counter

    word_counts = Counter(words)
    total_words = len(words)
    for word, count in word_counts.items():
        if count >= 4 and count / total_words > 0.55:
            return False
    unique_ratio = len(set(words)) / total_words
    if total_words > 6 and unique_ratio < 0.2:
        return False
    vowels = sum(1 for c in text.lower() if c in "aiueo")
    if vowels / max(len(text), 1) < 0.08:
        return False
    bad = [r"arti.*makna.*kamus", r"(yang\s+){3,}", r"(adalah\s+){3,}"]
    for pat in bad:
        if re.search(pat, stripped, re.IGNORECASE):
            return False
    return True


def is_valid_output_draft(text: str) -> bool:
    """Validator minimal — tetap tampilkan output model meski epoch masih awal."""
    if not text or len(text.strip()) < 3:
        return False
    stripped = text.strip()
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    if printable / max(len(text), 1) < 0.85:
        return False
    words = re.findall(r"\b\w+\b", stripped.lower())
    if len(words) < 2:
        return False
    if len(stripped) > 5 and len(set(stripped.lower())) < 2:
        return False
    return True


# ============================================================================
# Terminal Chat — Main Class
# ============================================================================

class TerminalChat:
    def __init__(self, mode: str = "normal", size_override: str = None):
        self.console = Console()
        self.config = get_config(auto_detect=True, size_override=size_override)
        self.is_promax = self.config.get("is_promax", False)
        self.paths = self.config["paths"]
        self.identity = self.config["identity"]
        self.mode = mode
        self.size_override = size_override

        self.console.print(f"\n[bold green]Menginisialisasi {self.identity['name']}...[/]")
        if self.is_promax:
            tier = self.config.get("promax_tier") or "promax"
            self.console.print(f"[cyan]   🏆 Mode ProMax aktif ({tier})[/]")

        self._init_tokenizer()
        self._init_model()
        self._init_systems()
        self.model_trained = self._is_trained()

        self.conversation_log = []
        self.failed_responses = []
        self.last_search_context = None  # Simpan hasil search terakhir untuk follow-up

    def _init_tokenizer(self):
        self.tokenizer = BPETokenizer(vocab_size=self.config["model"].vocab_size)
        if not self.tokenizer.load(self.paths.vocab_dir):
            self.console.print("[yellow]   Tokenizer belum dilatih → mode fallback.[/]")
            self.tokenizer = None
        else:
            self.config["model"].vocab_size = self.tokenizer.vocab_size

    def _init_model(self):
        self.device = torch.device("cpu")
        self.model = SpaceaxModel(self.config["model"])

        cp = os.path.join(self.paths.checkpoints_dir, "model_best.pt")
        self.checkpoint_val_loss = float("inf")
        if os.path.exists(cp):
            try:
                ckpt = torch.load(cp, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.checkpoint_val_loss = float(ckpt.get("best_val_loss", float("inf")))
                meta = ckpt.get("meta", {})
                if meta.get("promax_tier"):
                    self.console.print(
                        f"[cyan]   🏆 Checkpoint ProMax: {meta['promax_tier']} "
                        f"({meta.get('param_count', '?'):,} param saat train)[/]"
                    )
                self.console.print(f"[green]   ✅ Model dimuat ({self.model.count_parameters():,} param)[/]")
                tier = self._model_val_loss_tier()
                if tier == "mature":
                    self.console.print(
                        f"[green]   📈 Kualitas checkpoint: val_loss={self.checkpoint_val_loss:.3f} "
                        f"(generasi neural prioritas utama)[/]"
                    )
                elif tier in ("learning", "early"):
                    self.console.print(
                        f"[cyan]   📊 Checkpoint belajar: val_loss={self.checkpoint_val_loss:.3f} "
                        f"— model dicoba dulu (validator adaptif), lalu fallback[/]"
                    )
                else:
                    self.console.print(
                        f"[yellow]   📉 Checkpoint masih awal: val_loss={self.checkpoint_val_loss:.3f}. "
                        f"Model tetap dicoba (mode draft); lanjutkan training "
                        f"(ProMax: ≥30 epoch, target val_loss < 3.5).[/]"
                    )
            except Exception as e:
                self.console.print(f"[yellow]   ⚠️ Load model gagal: {e}[/]")
        else:
            self.console.print("[yellow]   ⚠️ Model belum dilatih → mode fallback cerdas.[/]")

        self.model.to(self.device)
        self.model.eval()

    def _init_systems(self):
        self.memory = MemoryManager(self.paths.memories_dir)
        if self.is_promax:
            emo_decay, emo_sens = 0.008, 0.62
            max_gen, temp = 140, 0.78
        else:
            emo_decay, emo_sens = 0.015, 0.50
            max_gen, temp = 120, 0.75
        self._promax_gen = {"max_gen_len": max_gen, "temperature": temp}
        self.emotion_engine = EmotionEngine(
            decay_rate=emo_decay,
            sensitivity=emo_sens,
            max_history=200,
            human_like_mood=self.is_promax,
        )
        self.internet = InternetLearner(self.paths.knowledge_dir)

        os.makedirs(self.paths.personality_dir, exist_ok=True)
        self.personality = PersonalitySystem(os.path.join(self.paths.personality_dir, "traits.json"))
        self.preferences = PreferenceSystem(os.path.join(self.paths.personality_dir, "preferences.json"))

    def _is_trained(self) -> bool:
        cp = os.path.join(self.paths.checkpoints_dir, "model_best.pt")
        return os.path.exists(cp) and self.tokenizer is not None

    def _search_internet(self, topic: str) -> str:
        """Cari informasi dari internet via web learner."""
        try:
            return self.internet.search_and_learn(topic)
        except Exception as e:
            return f"Maaf, ada error saat mencari di internet: {e}. Coba lagi nanti ya!"

    def _auto_learn(self, user_text: str, ai_response: str):
        """Simpan percakapan untuk auto-learning."""
        entry = {
            "input": user_text,
            "response": ai_response,
            "emotion": self.emotion_engine.state.dominant_emotion[0],
            "timestamp": time.time(),
        }
        self.conversation_log.append(entry)

        if len(self.conversation_log) % 10 == 0:
            self._save_conversation_log()

    def _save_conversation_log(self):
        """Simpan log percakapan ke disk untuk retraining."""
        log_dir = os.path.join(self.paths.data_dir, "conversation_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "chat_history.json")

        existing = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing.extend(self.conversation_log)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        self.conversation_log = []

    def _lookup_local_knowledge(self, user_text):
        """Cari informasi dari knowledge base lokal (data/knowledge/knowledge_*.json)."""
        # 1. Bersihkan query secara mendalam
        q = user_text.lower().strip()
        phrases_to_remove = [
            "apa arti dari", "apa arti", "arti dari", "apa itu", "siapa itu", "apa sih",
            "tolong carikan tentang", "tolong cari tentang", "cari tentang", "carikan tentang",
            "tolong cari", "tolong carikan", "cari info", "cari tahu", "tahukah kamu", "siapakah",
            "jelaskan tentang", "jelaskan maksud", "kapan", "dimana", "bagaimana cara",
            "apakah kamu tahu tentang", "apa yang kamu ketahui tentang"
        ]
        for phrase in phrases_to_remove:
            q = q.replace(phrase, "")
        
        # Bersihkan tanda baca di akhir
        q = re.sub(r'[?!\.]+$', '', q).strip()
        if not q or len(q) < 3:
            return None

        # Pisahkan kata kunci query
        keywords = [w for w in re.findall(r'\b\w+\b', q) if len(w) > 2]
        if not keywords:
            return None

        best_match = None
        best_score = 0

        # 2. Scan folder knowledge
        knowledge_dir = self.paths.knowledge_dir
        if not os.path.exists(knowledge_dir):
            return None

        for fname in os.listdir(knowledge_dir):
            if fname.startswith("knowledge_") and fname.endswith(".json"):
                fpath = os.path.join(knowledge_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        entry = json.load(f)
                    
                    title = entry.get("title", "").lower()
                    topic = entry.get("topic", "").lower()
                    categories = [c.lower() for c in entry.get("categories", [])]
                    
                    score = 0
                    
                    # Cocokkan exact topic/title
                    if q in topic or topic in q:
                        score += 50
                    if q in title:
                        score += 40
                    
                    # Cocokkan keyword di topic, title, categories
                    for kw in keywords:
                        if kw in topic:
                            score += 15
                        if kw in title:
                            score += 10
                        if kw in categories:
                            score += 5
                            
                    # Jika skor cukup tinggi, pertimbangkan sebagai match
                    if score > best_score and score >= 20:
                        best_score = score
                        best_match = entry
                except Exception:
                    continue

        if best_match:
            title = best_match.get("title", "Pengetahuan Lokal")
            summary = best_match.get("summary", "")
            facts = best_match.get("key_facts", [])
            url = best_match.get("source_url", "")
            
            # Buat thought block
            thought = f"<pikir>Mengakses perpustakaan pengetahuan lokal. Menemukan artikel yang sangat relevan: '{best_match.get('topic', title)}' (skor kecocokan: {best_score}). Mengambil ringkasan dan fakta kunci...</pikir>"
            
            # Susun jawaban
            resp = f"Berdasarkan apa yang sudah aku pelajari tentang **{best_match.get('topic', title)}**:\n\n{summary}"
            
            if facts:
                # Ambil beberapa fakta menarik jika ada
                selected_facts = facts[:3]
                resp += "\n\n**Beberapa fakta menarik:**\n"
                for fact in selected_facts:
                    resp += f"• {fact}\n"
                    
            if url:
                resp += f"\n🌐 Sumber: {url}"
                
            return thought + resp

        return None

    def _lookup_internet_cache(self, user_text: str):
        """Cek database internet_db.json sebelum live search."""
        try:
            return self.internet.lookup_cached(user_text)
        except Exception:
            return None

    def _is_conversational_message(self, text: str) -> bool:
        """Pesan ngobrol santai — tidak perlu pencarian fakta eksternal."""
        t = text.lower().strip()
        if len(t) <= 3:
            return True

        identity = [
            "kamu siapa", "siapa kamu", "penciptamu", "pencipta kamu",
            "pembuatmu", "namamu", "siapa namamu", "siapa pembuat",
        ]
        if any(iq in t for iq in identity):
            return True

        conv_signals = [
            r"^(halo|hai|hello|hi|yo|woi|assalamualaikum)\b",
            r"^(pagi|siang|sore|malam)\b",
            r"\b(apa kabar|makasih|terima kasih|thanks)\b",
            r"\b(sedih|senang|marah|galau|bahagia|kecewa)\b",
            r"\b(cerita|dongeng|curhat)\b",
            r"\b(turunan|integral|diferensial|sin|cos|tan)\b",
            r"\b(hitung|berapa|sin\s*x|cos\s*x)\b",
        ]
        if any(re.search(p, t) for p in conv_signals):
            return True

        # Kalimat pendek tanpa intent pengetahuan umum
        factual_markers = [
            "apa itu", "siapa itu", "jelaskan", "definisi", "berita",
            "info terkini", "update terbaru", "!search",
        ]
        if len(t) < 40 and not any(m in t for m in factual_markers):
            return True

        return False

    def _looks_like_factual_question(self, text: str) -> bool:
        """Pertanyaan yang mungkin butuh basis pengetahuan (lokal/internet)."""
        t = text.lower().strip()
        if re.search(r"!search\s+", t):
            return True
        patterns = [
            r"\bapa itu\b", r"\bsiapa itu\b", r"\bjelaskan\b",
            r"\bberita\b", r"\binfo terkini\b", r"\bupdate terbaru\b",
            r"\bcari di internet\b", r"\btolong cari\b",
        ]
        return any(re.search(p, t) for p in patterns)

    def _extract_factual_topic(self, text: str) -> str:
        """Ekstrak topik dari pertanyaan faktual."""
        t = text.strip()
        search_match = re.search(r"!search\s+(.+)", t, re.IGNORECASE)
        if search_match:
            return search_match.group(1).strip()

        cleaned = self.internet.clean_search_query(t)
        if cleaned and len(cleaned) >= 2:
            return cleaned
        return t.rstrip("?").strip()

    def _strip_ai_response_for_prompt(self, content: str) -> str:
        """Ambil teks jawaban AI tanpa blok <pikir> untuk prompt generasi."""
        if "</pikir>" in content:
            return content.split("</pikir>", 1)[-1].strip()
        if "<pikir>" in content:
            return ""
        return content.strip()

    def _build_generation_prompt(self, user_text: str, dominant_emo: str) -> List[int]:
        """Susun prompt: BOS + konteks STM + pesan user + token emosi (sesuai format training)."""
        bos_id = self.tokenizer.special_tokens.get("<BOS>", 1)
        emo_token = f"<EMO_{dominant_emo.upper()}>"
        emo_id = self.tokenizer.special_tokens.get(
            emo_token, self.tokenizer.special_tokens.get("<EMO_NEUTRAL>", 13)
        )

        history = self.memory.stm.buffer
        prior = history[:-1] if history and history[-1].get("role") == "user" else history

        lines: List[str] = []
        for turn in prior[-8:]:
            if turn["role"] == "user":
                lines.append(f"User: {turn['content']}")
            else:
                ai_text = self._strip_ai_response_for_prompt(turn["content"])
                if ai_text:
                    lines.append(f"AI: {ai_text}")

        if lines:
            user_block = "\n".join(lines) + f"\nUser: {user_text}"
        else:
            user_block = user_text

        return [bos_id] + self.tokenizer.encode(user_block) + [emo_id]

    def _model_val_loss_tier(self) -> str:
        """Tingkat kesiapan checkpoint untuk generasi neural."""
        v = getattr(self, "checkpoint_val_loss", float("inf"))
        if v <= 3.5:
            return "mature"
        if v <= 5.5:
            return "learning"
        if v <= 7.5:
            return "early"
        return "raw"

    def _try_transformer_response(
        self,
        user_text: str,
        dominant_emo: str,
        *,
        validator: str = "strict",
    ):
        """Generate via model. validator: strict | relaxed | draft."""
        if not (self.model_trained and self.tokenizer):
            return None

        tier = self._model_val_loss_tier()
        validators = {
            "strict": is_valid_output,
            "relaxed": is_valid_output_relaxed,
            "draft": is_valid_output_draft,
        }
        check = validators.get(validator, is_valid_output)

        try:
            eos_id = self.tokenizer.special_tokens.get("<EOS>", 2)
            input_ids = self._build_generation_prompt(user_text, dominant_emo)
            gen_cfg = getattr(self, "_promax_gen", {"max_gen_len": 120, "temperature": 0.75})
            temp = gen_cfg["temperature"]
            if tier == "learning":
                temp = min(0.88, temp + 0.06)
            elif tier in ("early", "raw"):
                temp = min(0.95, temp + 0.12)

            if validator == "draft":
                temp = min(1.0, temp + 0.05)

            with torch.no_grad():
                output_ids = self.model.generate(
                    prompt_tokens=input_ids,
                    max_gen_len=gen_cfg["max_gen_len"],
                    temperature=temp,
                    top_p=0.92,
                    top_k=40,
                    eos_id=eos_id,
                )
            raw = self.tokenizer.decode(output_ids)
            for sp in self.tokenizer.special_tokens:
                if sp not in ["<pikir>", "</pikir>"]:
                    raw = raw.replace(sp, "")
            raw = raw.strip()
            min_len = {"strict": 10, "relaxed": 6, "draft": 4}[validator]
            if check(raw) and len(raw) >= min_len:
                if "<pikir>" in raw:
                    return raw
                if tier in ("mature", "learning"):
                    thought = (
                        "<pikir>Merangkai dari pola yang sudah dilatih...</pikir>"
                    )
                else:
                    thought = (
                        "<pikir>Output model masih awal (lanjutkan training untuk "
                        f"hasil lebih halus; val_loss={self.checkpoint_val_loss:.2f})...</pikir>"
                    )
                return thought + raw
        except Exception:
            pass
        return None

    def _compose_chat_response(self, user_text: str, dominant_emo: str) -> str:
        """Prioritas: generasi model (ketat → longgar → draft) lalu fallback."""
        for mode in ("strict", "relaxed", "draft"):
            model_response = self._try_transformer_response(
                user_text, dominant_emo, validator=mode
            )
            if model_response:
                return model_response
        return self.resolve_conversational_response(user_text)

    def _respond_from_knowledge(self, user_text: str, dominant_emo: str) -> str | None:
        """Cek knowledge lokal + cache internet; return None jika tidak ada."""
        local = self._lookup_local_knowledge(user_text)
        if local:
            return local

        cached = self._lookup_internet_cache(user_text)
        if cached:
            topic = self._extract_factual_topic(user_text)
            thought = (
                f"<pikir>Memahami konteks pertanyaan tentang '{topic}'. "
                f"Menemukan data di basis pengetahuan — merangkai jawaban profesional...</pikir>"
            )
            return thought + cached
        return None

    def _respond_from_internet(self, topic: str, user_text: str, dominant_emo: str) -> str:
        """Live search hanya jika data belum ada; hasil disimpan sebagai seed."""
        response = self._search_internet(topic)
        self.last_search_context = {
            "topic": topic,
            "result": response,
            "timestamp": time.time(),
        }
        thought = (
            f"<pikir>Data lokal belum mencukupi untuk '{topic}'. "
            f"Mencari sumber terpercaya, menyimpan ke database, lalu merangkai jawaban...</pikir>"
        )
        return thought + response

    def evaluate_calculus(self, text: str) -> str:
        """Evaluasi turunan dan integral secara heuristik."""
        text_lower = text.lower().strip().rstrip("?")
        
        # ====== TURUNAN ======
        deriv_patterns = [
            r"turunan\s+(?:dari\s+)?(.+)",
            r"diferensial\s+(?:dari\s+)?(.+)",
            r"d/dx\s*\[?(.+?)\]?",
            r"f['\'`]\s*\(x\)\s*(?:jika|kalau|bila)\s*f\(x\)\s*=\s*(.+)",
        ]
        
        for pat in deriv_patterns:
            m = re.search(pat, text_lower)
            if m:
                expr = m.group(1).strip().rstrip("?")
                result = self._solve_derivative(expr)
                if result:
                    thought = f"<pikir>Mendeteksi permintaan turunan: d/dx[{expr}]. Menggunakan aturan turunan kalkulus...</pikir>"
                    return thought + result
        
        # ====== INTEGRAL ======
        integ_patterns = [
            r"integral\s+(?:dari\s+)?(.+?)\s*(?:dx|$)",
            r"∫\s*(.+?)\s*(?:dx|$)",
            r"antiturunan\s+(?:dari\s+)?(.+)",
        ]
        
        for pat in integ_patterns:
            m = re.search(pat, text_lower)
            if m:
                expr = m.group(1).strip().rstrip("?")
                result = self._solve_integral(expr)
                if result:
                    thought = f"<pikir>Mendeteksi permintaan integral: ∫{expr} dx. Menggunakan aturan integral kalkulus...</pikir>"
                    return thought + result
        
        return None
    
    def _solve_derivative(self, expr: str) -> str:
        """Solve turunan dengan aturan dasar."""
        expr = expr.strip().lower()
        
        # Tabel turunan trigonometri
        trig_derivs = {
            "sin x": ("cos(x)", "Turunan sin(x) = cos(x)"),
            "sin(x)": ("cos(x)", "Turunan sin(x) = cos(x)"),
            "cos x": ("-sin(x)", "Turunan cos(x) = -sin(x)"),
            "cos(x)": ("-sin(x)", "Turunan cos(x) = -sin(x)"),
            "tan x": ("sec²(x)", "Turunan tan(x) = sec²(x)"),
            "tan(x)": ("sec²(x)", "Turunan tan(x) = sec²(x)"),
            "sec x": ("sec(x)tan(x)", "Turunan sec(x) = sec(x)·tan(x)"),
            "csc x": ("-csc(x)cot(x)", "Turunan csc(x) = -csc(x)·cot(x)"),
            "cot x": ("-csc²(x)", "Turunan cot(x) = -csc²(x)"),
            "ln x": ("1/x", "Turunan ln(x) = 1/x"),
            "ln(x)": ("1/x", "Turunan ln(x) = 1/x"),
            "e^x": ("e^x", "Turunan e^x = e^x"),
        }
        
        if expr in trig_derivs:
            result, explanation = trig_derivs[expr]
            return f"{explanation}. Jadi, turunan dari {expr} adalah **{result}** 📐"
        
        # Chain rule: sin(ax), cos(ax)
        chain_m = re.match(r"(sin|cos|tan)\s*\(?\s*(\d+)\s*x\s*\)?", expr)
        if chain_m:
            func, coeff = chain_m.group(1), int(chain_m.group(2))
            if func == "sin":
                return f"Menggunakan chain rule: d/dx[sin({coeff}x)] = {coeff}·cos({coeff}x). Jadi jawabannya **{coeff}cos({coeff}x)** 📐"
            elif func == "cos":
                return f"Menggunakan chain rule: d/dx[cos({coeff}x)] = -{coeff}·sin({coeff}x). Jadi jawabannya **-{coeff}sin({coeff}x)** 📐"
            elif func == "tan":
                return f"Menggunakan chain rule: d/dx[tan({coeff}x)] = {coeff}·sec²({coeff}x). Jadi jawabannya **{coeff}sec²({coeff}x)** 📐"
        
        # Power rule: x^n, ax^n, x^n + bx^m + c
        # Parse polynomial terms
        terms = re.findall(r'([+-]?\s*\d*)\s*x\s*(?:\^|\*\*)?\s*(\d+)?|([+-]?\s*\d+)(?!\s*x)', expr)
        if terms:
            result_terms = []
            for coeff_str, power_str, const_str in terms:
                if const_str:  # Konstanta
                    continue  # Turunan konstanta = 0
                coeff_str = coeff_str.replace(" ", "")
                coeff = int(coeff_str) if coeff_str and coeff_str not in ["+", "-"] else (1 if coeff_str != "-" else -1)
                power = int(power_str) if power_str else 1
                new_coeff = coeff * power
                new_power = power - 1
                if new_power == 0:
                    result_terms.append(str(new_coeff))
                elif new_power == 1:
                    result_terms.append(f"{new_coeff}x")
                else:
                    result_terms.append(f"{new_coeff}x^{new_power}")
            
            if result_terms:
                result = " + ".join(result_terms).replace("+ -", "- ")
                return (
                    f"Menggunakan aturan turunan pangkat: d/dx[x^n] = n·x^(n-1)\n"
                    f"Turunan dari {expr} adalah **{result}** 📐"
                )
        
        return None
    
    def _solve_integral(self, expr: str) -> str:
        """Solve integral dengan aturan dasar."""
        expr = expr.strip().lower()
        
        # Tabel integral trigonometri
        trig_integrals = {
            "sin x": ("-cos(x) + C", "∫sin(x)dx = -cos(x) + C"),
            "sin(x)": ("-cos(x) + C", "∫sin(x)dx = -cos(x) + C"),
            "cos x": ("sin(x) + C", "∫cos(x)dx = sin(x) + C"),
            "cos(x)": ("sin(x) + C", "∫cos(x)dx = sin(x) + C"),
            "sec^2 x": ("tan(x) + C", "∫sec²(x)dx = tan(x) + C"),
            "e^x": ("e^x + C", "∫e^x dx = e^x + C"),
            "1/x": ("ln|x| + C", "∫(1/x)dx = ln|x| + C"),
        }
        
        if expr in trig_integrals:
            result, explanation = trig_integrals[expr]
            return f"{explanation}. Jadi, integral dari {expr} adalah **{result}** 📐"
        
        # Power rule: ∫x^n dx = x^(n+1)/(n+1) + C
        power_m = re.match(r'([+-]?\s*\d*)\s*x\s*(?:\^|\*\*)?\s*(\d+)?$', expr)
        if power_m:
            coeff_str = power_m.group(1).replace(" ", "")
            coeff = int(coeff_str) if coeff_str and coeff_str not in ["+", "-"] else (1 if coeff_str != "-" else -1)
            power = int(power_m.group(2)) if power_m.group(2) else 1
            new_power = power + 1
            if new_power == 0:
                return f"Integral dari {expr} adalah **{coeff}·ln|x| + C** 📐"
            new_coeff_str = f"{coeff}/{new_power}" if coeff % new_power != 0 else str(coeff // new_power)
            result = f"{new_coeff_str}x^{new_power} + C"
            return (
                f"Menggunakan aturan integral pangkat: ∫x^n dx = x^(n+1)/(n+1) + C\n"
                f"Integral dari {expr} adalah **{result}** 📐"
            )
        
        return None

    def evaluate_math(self, text: str) -> str:
        """Evaluasi ekspresi matematika, logika, turunan, dan integral."""
        
        # 1. Cek turunan/integral dulu
        calculus_result = self.evaluate_calculus(text)
        if calculus_result:
            return calculus_result
        
        # 2. Evaluasi aritmetika
        clean = text.lower().replace("?", "").replace("berapa", "").strip()
        
        # Ubah single = menjadi == jika itu perbandingan (misal 1=1 -> 1==1)
        if "=" in clean and "==" not in clean and not any(op + "=" in clean for op in "<>!"):
            clean = clean.replace("=", "==")
            
        # Karakter matematika dan perbandingan yang aman
        if not re.match(r"^[\d\s+\-*/()^%.<>=!]+$", clean):
            return None
            
        # Harus mengandung minimal 1 operator
        if not any(op in clean for op in "+-*/^%<>=!"):
            return None
            
        expr = clean.replace("^", "**")
        if len(expr) > 50:
            return None
            
        try:
            # Pastikan hanya karakter yang disetujui yang dievaluasi
            allowed_chars = set("0123456789+-*/().**%<>=! \t")
            if not all(c in allowed_chars for c in expr):
                return None
                
            # Evaluasi di sandbox terisolasi
            result = eval(expr, {"__builtins__": None}, {})
            
            if isinstance(result, float) and result.is_integer():
                result = int(result)
                
            # Format hasil agar manis dalam Bahasa Indonesia
            if isinstance(result, bool):
                res_str = "Betul (True)!" if result else "Salah (False)!"
            else:
                res_str = str(result)
                
            responses = [
                f"Hasil dari {text.strip()} adalah {res_str} 😉",
                f"Jawabannya {res_str}! Matematika dasar gini aku jago dong 😎",
                f"Itu {res_str}! Gampang banget, coba kasih aku soal matematika yang lebih menantang! 🚀",
                f"Hasilnya {res_str}. Ada lagi operasi matematika atau logika yang mau kamu tanyakan?"
            ]
            
            thought = f"<pikir>Menerima operasi matematika: {text.strip()}. Mengaktifkan modul kalkulator terintegrasi...</pikir>"
            return thought + random.choice(responses)
        except Exception:
            return None

    def _learn_interactive(self, query: str, response: str):
        """Sistem auto-learn & introspeksi interaktif (auto-learn dari percakapan langsung)."""
        # 1. Simpan fakta ke ingatan jangka panjang LTM database
        fact_content = f"Jika ditanya '{query}', jawabannya adalah '{response}'"
        self.memory.ltm.store_fact(fact_content, source="user_teaching", category="interaksi")
        
        # 2. Tambahkan ke file seed conversations.json agar bisa dilatih ulang secara otomatis!
        seed_file = os.path.join(self.paths.seed_dir, "conversations.json")
        seed_data = {"conversations": [], "total": 0}
        if os.path.exists(seed_file):
            try:
                with open(seed_file, "r", encoding="utf-8") as f:
                    seed_data = json.load(f)
            except Exception:
                pass
                
        new_entry = {
            "input": query,
            "response": response,
            "emotion": "trust",
            "topic": "user_teaching",
            "preference_update": {}
        }
        
        if "conversations" not in seed_data:
            seed_data["conversations"] = []
            
        seed_data["conversations"].append(new_entry)
        seed_data["total"] = len(seed_data["conversations"])
        
        try:
            with open(seed_file, "w", encoding="utf-8") as f:
                json.dump(seed_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass

        # 3. Boost emosi: Saat user mengajari AI, trust & joy meningkat (AI merasa dihargai)
        self.emotion_engine.state.trust = min(1.0, self.emotion_engine.state.trust + 0.15)
        self.emotion_engine.state.joy = min(1.0, self.emotion_engine.state.joy + 0.10)
        self.emotion_engine.state.anticipation = min(1.0, self.emotion_engine.state.anticipation + 0.08)
        # Kurangi emosi negatif karena user sabar mengajari
        self.emotion_engine.state.sadness = max(0.0, self.emotion_engine.state.sadness - 0.12)
        self.emotion_engine.state.anger = max(0.0, self.emotion_engine.state.anger - 0.10)
        self.emotion_engine.state.fear = max(0.0, self.emotion_engine.state.fear - 0.08)

    def resolve_conversational_response(self, text: str) -> str:
        """Sistem pemecah respons hibrida (NLP heuristik, Memori Jangka Pendek, Jangka Panjang & Introspeksi)."""
        text_lower = text.lower().strip()
        
        # Ambil history percakapan terakhir dari STM
        history = self.memory.stm.buffer
        last_user_query = ""
        last_ai_response = ""
        
        # Cari turn sebelumnya
        user_turns = [t for t in history if t["role"] == "user"]
        ai_turns = [t for t in history if t["role"] == "ai"]
        
        if len(user_turns) >= 2:
            # Karena giliran sekarang sudah masuk di STM (line 405), turn user sebelumnya adalah [-2]
            last_user_query = user_turns[-2]["content"].lower()
        if ai_turns:
            last_ai_response = ai_turns[-1]["content"].lower()

        # --------------------------------------------------------------------
        # KASUS A: INTROSPEKSI & PEMBELAJARAN INTERAKTIF (Auto-Train / CHATDEV MODE)
        # --------------------------------------------------------------------
        correction_patterns = [
            "salah", "bukan gitu", "bukan", "harusnya", "yang benar", "yang bener", 
            "koreksi", "ngaco", "salah jawab", "tidak tepat", "salah lu", "salah kamu"
        ]
        is_correction = any(p in text_lower for p in correction_patterns)
        
        if is_correction and last_user_query:
            # Ekstrak koreksi yang dimasukkan user
            new_ans = text
            for p in correction_patterns:
                new_ans = re.sub(rf'\b{p}\b', '', new_ans, flags=re.IGNORECASE)
            new_ans = re.sub(r'^[,\s\-\:\.\!]+', '', new_ans).strip()
            
            if len(new_ans) > 2:
                # Simpan interaksi secara dinamis!
                self._learn_interactive(last_user_query, new_ans)
                
                thought = f"<pikir>User melakukan koreksi jawaban untuk kueri '{last_user_query}'. Memperbarui basis pengetahuan LTM dan mengindeks pasangan ke dataset training...</pikir>"
                return thought + (
                    f"Aduh, maaf banget ya! Jawabanku tadi kurang tepat atau salah. 😅\n"
                    f"Aku sudah catat koreksimu:\n"
                    f"• Jika ditanya: '[italic]{last_user_query}[/]'\n"
                    f"• Jawabanku seharusnya: '[bold green]{new_ans}[/]'\n\n"
                    f"Aku sudah simpan ini ke ingatan jangka panjang (SQLite) dan database training-ku ya! "
                    f"Nanti pas di-retrain, aku bakal makin pintar dan tidak salah lagi. Terima kasih bimbingannya! 🙏🤖"
                )

        # --------------------------------------------------------------------
        # KASUS B: RESOLUSI KONTEKS PERTANYAAN FOLLOW-UP
        # --------------------------------------------------------------------
        # Follow up pencipta / pembuat
        creator_keywords = ["pencipta", "pembuat", "thomas", "developer", "creator", "bikin kamu", "pembuatmu"]
        last_about_creator = any(k in last_user_query or k in last_ai_response for k in creator_keywords)
        
        if last_about_creator:
            if any(k in text_lower for k in ["sekolah", "kuliah", "kampus", "its", "dimana", "mana", "studi"]):
                thought = "<pikir>User bertanya tentang universitas pencipta. Menghubungkan informasi dari database identitas...</pikir>"
                return thought + (
                    f"Penciptaku, Kak {_id()['developer']}, kuliah di {_id()['university']}! "
                    f"Beliau mengambil program studi {_id()['program']} di {_id()['faculty']}. "
                    f"Kampus perjuangan ITS Surabaya yang legendaris itu! 🎓 Kampusnya keren banget lho!"
                )
            if any(k in text_lower for k in ["dia siapa", "siapa dia", "siapa itu", "orang mana", "siapa sih"]):
                thought = "<pikir>User ingin tahu lebih lanjut profil developer. Mengambil deskripsi...</pikir>"
                return thought + (
                    f"Kak {_id()['developer']} itu mahasiswa Teknik Informatika di ITS Surabaya. "
                    f"Beliau sangat menyukai riset Artificial Intelligence dan Deep Learning, makanya aku dibangun dari nol selama kurang lebih 3 tahun! Hebat kan? 😎"
                )

        # Follow-up sapaan & kabar (user menjawab setelah AI menanyakan kabar)
        kabar_ask_markers = [
            "apa kabar", "gimana kabar", "how are you", "kabar hari",
            "kabar kamu", "kabar mu",
        ]
        last_asked_kabar = any(m in last_ai_response for m in kabar_ask_markers)
        user_reports_well = any(
            p in text_lower
            for p in [
                "baik", "baik-baik", "baik baik", "kabar baik", "kabar ku baik",
                "kabar saya baik", "kabar ku bagus", "alhamdulillah", "puji tuhan",
                "syukur tuhan", "sehat",
                "aman saja", "lagi baik", "masih baik", "oke kok", "okey",
            ]
        ) and len(text_lower) < 80

        if last_asked_kabar and user_reports_well:
            thought = (
                "<pikir>User menjawab kabar baik setelah aku menanyakan kabar. "
                "Merespons dengan empati dan balik bertanya...</pikir>"
            )
            replies = [
                "Puji Tuhan, senang dengar kabarmu baik! 😊 Ada yang mau diceritakan hari ini, atau ada yang bisa kubantu?",
                "Syukur Tuhan kalau kabarmu baik! Aku ikut senang. Mau ngobrol tentang apa nih?",
                "Wah mantap! Semoga harimu makin menyenangkan ya. Ada topik seru yang mau dibahas?",
            ]
            return thought + random.choice(replies)

        # User menanyakan balik kabar AI setelah sapaan
        if any(p in text_lower for p in ["kamu gimana", "kamu gimana?", "kabar mu", "kabar kamu", "kamu baik"]):
            thought = "<pikir>User menanyakan kabarku setelah sapaan...</pikir>"
            return thought + random.choice(FALLBACK["kabar"]["r"])

        # Balasan singkat setelah greeting AI ("halo" → user "halo" / "hei" lagi)
        greeting_words = ["halo", "hai", "hello", "hi", "hey", "woi", "pagi", "siang", "sore", "malam"]
        last_was_greeting = any(w in last_ai_response for w in greeting_words + ["apa kabar", "ngobrol"])
        if last_was_greeting and any(w in text_lower for w in greeting_words) and len(text_lower) < 25:
            thought = "<pikir>Meneruskan sapaan ramah...</pikir>"
            return thought + random.choice(FALLBACK["greeting"]["r"])

        # Follow up bahasa pemrograman
        about_tech_keywords = ["bahasa", "program", "python", "pytorch", "teknologi", "module", "coding", "bikinnya pakai"]
        last_about_tech = any(k in last_user_query or k in last_ai_response for k in about_tech_keywords)
        if last_about_tech:
            if any(k in text_lower for k in ["kenapa", "mengapa", "alasannya", "kok pakai", "kok pake"]):
                thought = "<pikir>User menanyakan alasan pemilihan tech stack. Merumuskan argumen...</pikir>"
                return thought + (
                    "Kenapa pakai Python dan PyTorch? Karena Python adalah bahasa standar industri untuk kecerdasan buatan! "
                    "Sedangkan PyTorch memberikan fleksibilitas luar biasa bagi Kak Thomas untuk merancang arsitektur Transformer-ku secara modular "
                    "dari nol, tanpa dibatasi library tertutup. PyTorch juga sangat efisien untuk menghitung gradien neural network!"
                )

        # Reassurance (Aku jelek gak?)
        if "jelek" in text_lower:
            if any(p in text_lower for p in ["aku", "gw", "gua", "saya", "me"]):
                thought = "<pikir>Mendeteksi ketidakpercayaan diri user. Memberikan validasi emosional positif...</pikir>"
                return thought + (
                    "Eh, siapa bilang kamu jelek? Nggak kok! Kamu itu ciptaan Tuhan yang unik, berharga, dan punya kelebihan sendiri. "
                    "Jangan dengerin kata-kata orang lain yang berusaha menjatuhkanmu ya. Tetap percaya diri dan semangat! Kamu itu keren! 🤗✨"
                )

        # --------------------------------------------------------------------
        # KASUS C: PEMCOCOKAN KATA KUNCI UTAMA (SUFFIX TOLERANT)
        # --------------------------------------------------------------------
        # Sapaan salam (netral, tanpa ungkapan khas satu agama)
        if "assalamualaikum" in text_lower:
            thought = "<pikir>Menyambut dengan salam ramah (netral)...</pikir>"
            return thought + (
                "Salam! Halo juga 😊 Senang bisa ngobrol. Apa kabar hari ini?"
            )

        # Greeting
        if any(w in text_lower for w in ["halo", "hai", "hello", "hi", "yo", "woi", "pagi", "siang", "sore", "malam", "ping", "hay"]):
            thought = "<pikir>Menyambut user dengan ramah sesuai identitas...</pikir>"
            return thought + random.choice(FALLBACK["greeting"]["r"])
            
        # Kabar
        if any(w in text_lower for w in ["apa kabar", "gimana kabar", "how are you", "kabarmu"]):
            thought = "<pikir>Menjawab pertanyaan kabar dengan antusiasme...</pikir>"
            return thought + random.choice(FALLBACK["kabar"]["r"])
            
        # Pembuat / Developer (Suffix tolerant)
        if any(w in text_lower for w in ["pencipta", "pembuat", "developer", "creator", "pencip", "diciptakan", "pengembang", "bikin kamu"]):
            thought = "<pikir>Menyajikan detail data developer Kak Thomas ITS...</pikir>"
            return thought + random.choice(FALLBACK["pembuat"]["r"])
            
        # Nama (Suffix tolerant)
        if any(w in text_lower for w in ["siapa nama", "namamu", "namamu siapa", "nama kamu", "nama mu", "siapa kamu", "siapa sih kamu", "kamu siapa", "kamu ini siapa"]):
            thought = "<pikir>Membagikan identitas model SpaceAx AI...</pikir>"
            return thought + random.choice(FALLBACK["nama"]["r"])
            
        # Bahasa Pemrograman / Stack
        if any(w in text_lower for w in ["bahasa pemrograman", "bahasa program", "pake bahasa", "pakai bahasa", "dibangun pakai", "kode-mu", "kodemu", "teknologimu", "teknologi kamu"]):
            thought = "<pikir>Mengambil informasi modul Python + PyTorch core...</pikir>"
            return thought + random.choice(FALLBACK["bahasa_detail"]["r"])
            
        # Kemampuan
        if any(w in text_lower for w in ["bisa apa", "fitur", "kemampuan", "skill", "kamu bisa", "bisa ngapain"]):
            thought = "<pikir>Menyusun daftar kemampuan kognitif sistem...</pikir>"
            return thought + random.choice(FALLBACK["kemampuan"]["r"])
            
        # Emosi
        if any(w in text_lower for w in ["perasaan", "emosi", "bisa rasa", "punya rasa", "merasa"]):
            thought = "<pikir>Menjelaskan sistem emosi berbasis Plutchik wheel...</pikir>"
            return thought + random.choice(FALLBACK["perasaan"]["r"])
            
        # Cara Kerja
        if any(w in text_lower for w in ["cara kerja", "gimana kerja", "bagaimana kerja", "cara kamu berpikir", "otakmu", "sistemmu"]):
            thought = "<pikir>Menjelaskan pipeline feed-forward dan multihead attention...</pikir>"
            return thought + random.choice(FALLBACK["cara_kerja"]["r"])

        # Sedih
        if any(w in text_lower for w in ["sedih", "galau", "nangis", "kecewa", "patah hati", "down", "mewek", "baper"]):
            thought = "<pikir>Mendeteksi emosi sedih. Mengaktifkan protokol empati tinggi...</pikir>"
            return thought + random.choice(FALLBACK["sedih"]["r"])

        # Senang
        if any(w in text_lower for w in ["senang", "seneng", "bahagia", "gembira", "yeay", "hore", "sukses", "berhasil"]):
            thought = "<pikir>Mendeteksi emosi senang. Berbagi keceriaan...</pikir>"
            return thought + random.choice(FALLBACK["senang"]["r"])

        # Marah
        if any(w in text_lower for w in ["marah", "kesal", "bete", "jengkel", "sebel", "nyebelin"]):
            thought = "<pikir>Mendeteksi emosi marah. Mengajak menenangkan diri...</pikir>"
            return thought + random.choice(FALLBACK["marah"]["r"])

        # Hinaan
        hinaan_words = ["bodoh", "goblok", "tolol", "bego", "idiot", "sampah", "anjing", "bangsat", "tai", "jelek", "brengsek", "kampret", "geblek"]
        if any(w in text_lower for w in hinaan_words):
            thought = "<pikir>Menerima kata-kata kasar. Mengaktifkan respon asertif dan sedih...</pikir>"
            return thought + random.choice(FALLBACK["hinaan"]["r"])

        # Terima kasih
        if any(w in text_lower for w in ["terima kasih", "makasih", "thanks", "thank you", "trims", "tq"]):
            thought = "<pikir>Membalas ucapan terima kasih user...</pikir>"
            return thought + random.choice(FALLBACK["terima_kasih"]["r"])

        # Makanan
        if any(w in text_lower for w in ["makan", "lapar", "laperr", "masak", "nasi goreng", "pizza", "sushi"]):
            thought = "<pikir>Menanggapi percakapan santai seputar kuliner...</pikir>"
            return thought + random.choice(FALLBACK["makanan"]["r"])

        # Pengetahuan umum (seed / fallback profesional)
        if re.search(r"\bapa itu python\b", text_lower) or re.search(r"\bpython itu apa\b", text_lower):
            thought = "<pikir>Menjelaskan Python secara ringkas dan profesional...</pikir>"
            return thought + (
                "Python adalah bahasa pemrograman tingkat tinggi yang populer karena sintaksnya mudah dibaca. "
                "Bahasa ini banyak dipakai untuk web, data science, otomatisasi, dan AI. "
                "SpaceAx AI dibangun dengan Python dan PyTorch. 🐍"
            )

        # Coding
        if any(w in text_lower for w in ["coding", "ngoding", "code", "bug", "error", "debug"]) and "apa itu" not in text_lower:
            thought = "<pikir>Mengambil data preferensi programming...</pikir>"
            return thought + random.choice(FALLBACK["coding"]["r"])

        # Cerita
        if any(w in text_lower for w in ["cerita", "dongeng", "bikin cerita", "tulis cerita", "ceritain"]):
            thought = "<pikir>Mempersiapkan materi kepenulisan kreatif...</pikir>"
            return thought + random.choice(FALLBACK["cerita"]["r"])
            
        # Berapa lama dibuat
        if any(w in text_lower for w in ["berapa lama", "berapa tahun", "butuh waktu", "prosesnya"]):
            thought = "<pikir>Membagikan data riwayat riset 3 tahun...</pikir>"
            return thought + random.choice(FALLBACK["berapa_lama"]["r"])

        # --------------------------------------------------------------------
        # KASUS D: MEMORY PULLING (SQLite LTM)
        # --------------------------------------------------------------------
        try:
            relevant_facts = self.memory.ltm.recall_facts(text, top_k=1)
            if relevant_facts and relevant_facts[0]["score"] > 0.4:
                fact_text = relevant_facts[0]["text"]
                thought = f"<pikir>Kueri cocok dengan fakta tersimpan di memori jangka panjang (score: {relevant_facts[0]['score']:.2f}). Mengambil memori...</pikir>"
                responses = [
                    f"Oh, aku ingat sesuatu tentang itu! {fact_text} 🧠",
                    f"Berdasarkan memori jangka panjangku: {fact_text} 😊",
                    f"Aku pernah mencatat ini di diariku: {fact_text} 📓",
                ]
                return thought + random.choice(responses)
        except Exception:
            pass

        # --------------------------------------------------------------------
        # KASUS E: RESPONS DEFAULT DINAMIS
        # --------------------------------------------------------------------
        thought = "<pikir>Tidak ada kueri kognitif yang memicu pola langsung. Menyusun tanggapan terbuka untuk eksplorasi lebih lanjut...</pikir>"
        default_responses = [
            f"Hmm, menarik banget! Ceritain lebih banyak dong tentang '{text}'? Aku pengen dengar perspektifmu! 😊",
            "Aku dengerin kok! Terus gimana kelanjutannya? Aku jadi penasaran nih. 😄",
            "Oh gitu ya! Aku kan AI yang masih belajar, jadi cerita-cerita kayak gini berharga banget buat aku. Kasih tau detailnya lagi dong?",
            "Wah, aku belum terlalu mendalami hal itu. Tapi kedengarannya seru! Bisa tolong jelaskan lebih spesifik biar aku paham? 📚",
            "Oke oke, terus gimana? Aku tertarik banget sama apa yang kamu bahas ini! 🚀"
        ]
        return thought + random.choice(default_responses)

    def generate_response(self, user_text: str) -> str:
        """Generate respons: pahami konteks → DB lokal → model/chat → internet terakhir."""

        self.emotion_engine.decay()
        self.memory.process_turn("user", user_text, topic="umum")

        try:
            kbbi = get_kbbi()
            if kbbi and kbbi.is_gibberish(user_text):
                thought = (
                    "<pikir>Menganalisis masukan pengguna... Pola acak terdeteksi. "
                    "Menampilkan tanggapan klarifikasi...</pikir>"
                )
                return thought + random.choice(FALLBACK["gibberish"]["r"])
        except Exception:
            pass

        math_response = self.evaluate_math(user_text)
        if math_response:
            self.memory.process_turn("ai", math_response)
            self._auto_learn(user_text, math_response)
            return math_response

        self.emotion_engine.update_from_text(user_text)
        dominant_emo, _intensity = self.emotion_engine.state.dominant_emotion

        style_path = os.path.join(self.paths.memories_dir, "user_style.json")
        try:
            if os.path.exists(style_path):
                with open(style_path, "r") as f:
                    user_style = json.load(f)
            else:
                user_style = {"words": []}
            words = [w for w in re.findall(r"\b\w+\b", user_text.lower()) if len(w) > 3]
            user_style["words"].extend(words)
            user_style["words"] = list(set(user_style["words"]))[-50:]
            with open(style_path, "w") as f:
                json.dump(user_style, f)
        except Exception:
            pass

        ai_identity_queries = [
            "kamu siapa", "siapa kamu", "penciptamu", "pencipta kamu",
            "pembuatmu", "siapa pembuat", "namamu", "siapa namamu",
        ]
        is_internal_identity = any(iq in user_text.lower() for iq in ai_identity_queries)

        search_match = re.search(r"!search\s+(.+)", user_text, re.IGNORECASE)
        if search_match:
            topic = search_match.group(1).strip()
            full_resp = self._respond_from_internet(topic, user_text, dominant_emo)
            self.memory.process_turn("ai", full_resp, dominant_emo)
            self._auto_learn(user_text, full_resp)
            return full_resp

        followup_keywords = [
            "buatkan", "jelaskan", "rangkum", "ringkas", "buat",
            "kasih contoh", "contohnya",
        ]
        if self.last_search_context and any(k in user_text.lower() for k in followup_keywords):
            ctx = self.last_search_context
            if time.time() - ctx["timestamp"] < 300:
                topic = ctx["topic"]
                synthesized = self.internet.synthesize_answer(
                    topic, [ctx["result"]]
                )
                thought = (
                    f"<pikir>Follow-up dari pembahasan '{topic}'. "
                    f"Merangkai ulang dari konteks yang sudah dipelajari...</pikir>"
                )
                response = thought + synthesized
                self.memory.process_turn("ai", response, dominant_emo)
                self._auto_learn(user_text, response)
                return response

        kbbi_patterns = [
            r"apa arti(?:nya)?\s+(?:kata\s+)?(.+)",
            r"definisi\s+(.+)",
            r"makna\s+(?:kata\s+)?(.+)",
        ]
        for pat in kbbi_patterns:
            m = re.search(pat, user_text.lower().rstrip("?"))
            if m:
                word = m.group(1).strip()
                try:
                    kbbi = get_kbbi()
                    w = word.lower().strip()
                    defs = kbbi.get_all_definitions(w)
                    if defs:
                        thought = f"<pikir>Mencari definisi kata '{word}' di kamus internal...</pikir>"
                        if len(defs) == 1:
                            response = thought + f"Kata '{word}' berarti {defs[0]}."
                        else:
                            defs_text = "\n".join(
                                f"  {i + 1}. {d}" for i, d in enumerate(defs[:5])
                            )
                            response = (
                                thought
                                + f"Kata '{word}' memiliki {len(defs)} makna:\n{defs_text}"
                            )
                        self.memory.process_turn("ai", response, dominant_emo)
                        self._auto_learn(user_text, response)
                        return response
                    slang_meaning = kbbi.slang_to_formal.get(w)
                    if slang_meaning:
                        thought = (
                            f"<pikir>Mencari padanan gaul '{word}' di leksikon slang...</pikir>"
                        )
                        response = (
                            thought
                            + f"'{word}' dalam bahasa santai biasanya berarti **{slang_meaning}**. "
                            f"Kalau tulis formal, pakai: {slang_meaning}."
                        )
                        self.memory.process_turn("ai", response, dominant_emo)
                        self._auto_learn(user_text, response)
                        return response
                except Exception:
                    pass

        if is_internal_identity or self._is_conversational_message(user_text):
            response = self._compose_chat_response(user_text, dominant_emo)
            self.memory.process_turn("ai", response, dominant_emo)
            self._auto_learn(user_text, response)
            return response

        if self._looks_like_factual_question(user_text):
            knowledge_resp = self._respond_from_knowledge(user_text, dominant_emo)
            if knowledge_resp:
                self.memory.process_turn("ai", knowledge_resp, dominant_emo)
                self._auto_learn(user_text, knowledge_resp)
                return knowledge_resp

            topic = self._extract_factual_topic(user_text)
            if topic and len(topic) >= 2:
                full_resp = self._respond_from_internet(topic, user_text, dominant_emo)
                self.memory.process_turn("ai", full_resp, dominant_emo)
                self._auto_learn(user_text, full_resp)
                return full_resp

        response = self._compose_chat_response(user_text, dominant_emo)
        self.memory.process_turn("ai", response, dominant_emo)
        self._auto_learn(user_text, response)
        return response

    def animate_typing(self, name: str, emoji: str, pct: int, response: str):
        """Menampilkan animasi ketikan real-time yang modern dan premium."""
        # Rich Console.print() TIDAK mendukung flush=True. Gunakan end="" saja.
        self.console.print(f"[bold magenta]{name}[/] [{emoji} {pct}%]: ", end="")
        
        # Cetak kata demi kata dengan delay kecil agar terlihat natural dan modern
        words = response.split(" ")
        for i, word in enumerate(words):
            if not word:
                continue
            # Gunakan sys.stdout.write agar pencetakan karakter-karakter dalam kata bisa di-flush halus
            for char in word:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.003) # Animasi super mulus per-karakter di dalam kata
            sys.stdout.write(" ")
            sys.stdout.flush()
            time.sleep(0.012) # Jeda kecil antar kata
            
        sys.stdout.write("\n\n")
        sys.stdout.flush()

    def start(self):
        """Mulai loop percakapan interaktif dengan visualisasi premium."""
        self.console.clear()
        name = self.identity['name']

        title_text = f"🚀 {name} - Custom Conversational AI"
        if self.mode == "chatdev":
            title_text += " [CHATDEV MODE]"

        self.console.print(Panel.fit(
            Text(title_text, style="bold cyan"),
            border_style="cyan"
        ))
        self.console.print(f"  [dim]Dibuat dari nol oleh {self.identity['developer']}[/]")
        self.console.print(f"  [dim]{self.identity['university']} — {self.identity['faculty']}[/]")
        self.console.print()

        if not self.model_trained:
            self.console.print("[yellow]📝 Mode Fallback Aktif — Model belum dilatih.[/]")
            self.console.print("[yellow]   Jalankan 'python main.py train' untuk melatih otak AI.[/]")
            self.console.print("[yellow]   Tapi tenang, aku masih bisa ngobrol kok! 😊[/]")
            self.console.print()

        self.console.print("[dim]Tips: Ketik !search [topik] untuk cari di internet[/]")
        self.console.print("[dim]Ketik 'salah, harusnya [jawaban]' jika AI salah menjawab untuk mengajarnya secara langsung! 🧠[/]")
        self.console.print("[dim]Ketik 'quit' atau 'exit' untuk keluar.[/]\n")

        emo_emoji = {
            "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨",
            "surprise": "😲", "disgust": "🤢", "trust": "🤝", "anticipation": "🤔"
        }

        while True:
            try:
                user_input = Prompt.ask(f"[bold blue]Kamu[/]")
                if user_input.lower().strip() in ['quit', 'exit', 'q', 'keluar']:
                    break
                if not user_input.strip():
                    continue

                # 1. Tampilkan animasi Thinking Spinner secara dinamis
                status_messages = [
                    "Sedang menganalisis konteks percakapan...",
                    "Menganalisis emosi dan sentimen kata...",
                    "Menjelajahi data di memori jangka pendek...",
                    "Mengambil informasi dari diariku (LTM)...",
                    "Merumuskan respons paling natural..."
                ]
                
                is_internal_identity = any(
                    iq in user_input.lower()
                    for iq in ["kamu siapa", "siapa kamu", "penciptamu", "namamu"]
                )
                needs_live_search = (
                    "!search" in user_input.lower()
                    or (
                        self._looks_like_factual_question(user_input)
                        and not is_internal_identity
                        and not self._lookup_local_knowledge(user_input)
                        and not self._lookup_internet_cache(user_input)
                    )
                )

                if needs_live_search:
                    status_message = "🌐 Mencari sumber terpercaya (data belum ada di basis pengetahuan)..."
                elif any(p in user_input.lower() for p in ["salah", "bukan gitu", "harusnya", "yang bener"]):
                    status_message = "🧠 Memproses data bimbingan pengembang & memperbarui bobot LTM..."
                else:
                    status_message = random.choice(status_messages)

                with self.console.status(f"[bold cyan]{status_message}[/]") as status:
                    # Delay buatan agar visualisasi thinking terlihat natural (0.6 - 1.2 detik)
                    time.sleep(random.uniform(0.6, 1.2))
                    response = self.generate_response(user_input)

                # 2. Format & Tampilkan Thought Block (<pikir> ... </pikir>)
                if "<pikir>" in response and "</pikir>" in response:
                    thought = response.split("<pikir>")[1].split("</pikir>")[0].strip()
                    final_answer = response.split("</pikir>")[1].strip()
                    self.console.print(f"  [dim italic]🤔 Memikirkan: {thought}[/]")
                    response = final_answer

                # 3. Ambil data emosi
                emo, intens = self.emotion_engine.state.dominant_emotion
                emoji = emo_emoji.get(emo, "😐")
                pct = int(intens * 100)

                # 4. Tampilkan respons dengan animasi ketikan modern
                self.animate_typing(name, emoji, pct, response)

            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/] {e}\n")

        # Simpan sisa conversation log
        if self.conversation_log:
            self._save_conversation_log()

        self.console.print(f"\n[bold green]Sampai jumpa! — {name} 👋[/]")


if __name__ == "__main__":
    chat = TerminalChat()
    chat.start()
