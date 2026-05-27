import json
import os
import requests
import re
import warnings
# Diamkan warning renaming dari duckduckgo_search
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

from bs4 import BeautifulSoup
from ddgs import DDGS
from datetime import datetime, timezone

class InternetLearner:
    # Domain tepercaya yang diberi prioritas lebih tinggi
    TRUSTED_DOMAINS = [
        "wikipedia.org", "id.wikipedia.org",
        "kompas.com", "detik.com", "liputan6.com",
        "tribunnews.com", "cnnindonesia.com", "tempo.co",
        "bbc.com", "bbc.co.uk", "britannica.com",
    ]

    # Kata kunci spam: citation generators, paraphrase tools, dll.
    SPAM_KEYWORDS = [
        # Citation / bibliography tools
        "citation", "bibliography", "apa style", "mla style", "chicago style",
        "apa format", "mla format", "apa citation", "mla citation",
        "cite this", "citation generator", "citation machine",
        "works cited", "in-text citation", "reference list",
        # Paraphrase / rewrite tools
        "grammarly", "plagiarism", "paraphrase", "paraphrasing",
        "rewrite", "rewriter", "turnitin", "quillbot", "wordtune",
        "plagiarism checker", "grammar checker",
        # Ad / sponsored
        "ads", "sponsored", "adwords",
    ]

    # Domain spam yang langsung diblokir
    SPAM_DOMAINS = [
        "scribbr.com", "easybib.com", "bibme.com", "citationmachine.net",
        "citethisforme.com", "mybib.com", "citefast.com",
        "formatically.com", "grafiati.com", "zbib.org",
        "quillbot.com", "grammarly.com", "turnitin.com", "wordtune.com",
    ]

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir
        self.db_path = os.path.join(knowledge_dir, "internet_db.json")
        self.knowledge_base = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_db(self):
        os.makedirs(self.knowledge_dir, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)

    def clean_search_query(self, query: str) -> str:
        """Membersihkan query pencarian dari kata tanya percakapan agar hasil search lebih presisi."""
        q = query.lower().strip()
        # Bersihkan awalan !search
        q = re.sub(r'^!search\s+', '', q)
        # Bersihkan kata tanya percakapan yang bisa mengotori hasil SEO
        phrases_to_remove = [
            "apa arti dari", "apa arti", "arti dari", "apa itu", "siapa itu", "apa sih",
            "tolong carikan tentang", "tolong cari tentang", "cari tentang", "carikan tentang",
            "tolong cari", "tolong carikan", "cari info", "cari tahu", "tahukah kamu", "siapakah",
            "jelaskan tentang", "jelaskan maksud"
        ]
        for phrase in phrases_to_remove:
            q = q.replace(phrase, "")
        
        # Bersihkan tanda tanya atau tanda seru di akhir
        q = re.sub(r'[?!\.]+$', '', q)
        return q.strip()

    def _is_spam(self, href: str, body: str) -> bool:
        """Cek apakah sebuah hasil pencarian termasuk spam/tidak relevan."""
        href_lower = href.lower()
        body_lower = body.lower()

        # Cek domain spam
        for domain in self.SPAM_DOMAINS:
            if domain in href_lower:
                return True

        # Cek kata kunci spam di URL atau body
        for keyword in self.SPAM_KEYWORDS:
            if keyword in href_lower or keyword in body_lower:
                return True

        return False

    def _score_result(self, result: dict, query_keywords: list) -> int:
        """Beri skor pada hasil pencarian berdasarkan relevansi dan kepercayaan sumber."""
        score = 0
        href = result.get("href", "").lower()
        body = result.get("body", "").lower()
        title = result.get("title", "").lower()

        # Bonus besar jika dari domain tepercaya
        for domain in self.TRUSTED_DOMAINS:
            if domain in href:
                score += 20
                break

        # Bonus jika body mengandung kata kunci pencarian
        for kw in query_keywords:
            if kw in body:
                score += 5
            if kw in title:
                score += 3

        # Bonus jika body cukup panjang (konten substantif)
        body_len = len(result.get("body", ""))
        if body_len > 150:
            score += 5
        elif body_len > 80:
            score += 2

        return score

    def search_and_learn(self, query: str) -> str:
        """Cari di internet secara cerdas, saring spam, dan kembalikan ringkasan terstruktur."""
        # Bersihkan query
        cleaned_query = self.clean_search_query(query)
        if not cleaned_query:
            cleaned_query = query
            
        query_key = query.lower().strip()
        
        # Cek cache lokal
        if query_key in self.knowledge_base:
            return self.knowledge_base[query_key]["summary"]

        print(f"\n[🌐 AI mengakses Internet untuk: '{cleaned_query}']...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with DDGS() as ddgs:
                    # Ambil 8 hasil agar punya banyak opsi setelah penyaringan
                    raw_results = list(ddgs.text(cleaned_query, max_results=8, region='id-id'))
            
            if not raw_results:
                return "Maaf, aku tidak menemukan informasi relevan di internet saat ini."

            # Saring hasil dari domain spam / ad / citation generator
            filtered_results = []
            
            for r in raw_results:
                href = r.get("href", "")
                body = r.get("body", "")
                # Jika link atau deskripsi berisi kata kunci spam, lewati
                if self._is_spam(href, body):
                    continue
                filtered_results.append(r)
                
            # Jika semua disaring, gunakan hasil mentah asal tidak kosong
            if not filtered_results:
                filtered_results = raw_results[:2]

            # Buat kata kunci dari query untuk scoring
            query_keywords = [w for w in cleaned_query.lower().split() if len(w) > 2]

            # Urutkan berdasarkan skor relevansi (tertinggi dulu)
            filtered_results.sort(key=lambda r: self._score_result(r, query_keywords), reverse=True)

            # Ambil maksimal 3 hasil terbaik
            results = filtered_results[:3]
            
            # Buat ringkasan yang bersih dan sertakan sumber sebagai link markdown
            summaries = []
            sources = []
            
            for i, r in enumerate(results):
                title = r.get("title", "Sumber Informasi")
                href = r.get("href", "")
                body = r.get("body", "").strip()
                
                # Tambahkan body snippet ke ringkasan
                if body:
                    summaries.append(body)
                if href:
                    # Buat format sumber yang rapi
                    clean_title = re.sub(r'[^\w\s\-]', '', title)[:40].strip()  # potong judul panjang
                    sources.append(f"[{clean_title}]({href})")
            
            # Hubungkan teks ringkasan secara alami
            combined_text = " ".join(summaries)
            
            # Jika terlalu panjang, potong pada batas kalimat terakhir agar tidak terputus kaku
            max_len = 500
            if len(combined_text) > max_len:
                truncated = combined_text[:max_len]
                # Cari titik akhir kalimat terdekat agar tidak terpotong di tengah kata/kalimat
                last_dot = truncated.rfind(".")
                if last_dot > 100:
                    summary_text = truncated[:last_dot + 1]
                else:
                    summary_text = truncated + "..."
            else:
                summary_text = combined_text

            # Gabungkan dengan sumber rujukan — format baru yang lebih rapi
            source_links = ", ".join(sources)
            summary = f"Berdasarkan informasi terbaru dari internet:\n\n{summary_text}\n\n🌐 Sumber: {source_links}"
            
            # Simpan ke cache lokal
            self.knowledge_base[query_key] = {
                "query": query,
                "summary": summary,
                "sources": [r.get("href", "") for r in results],
                "learned_at": datetime.now(timezone.utc).isoformat()
            }
            self._save_db()
            
            return summary
            
        except Exception as e:
            print(f"[Error Internet] {e}")
            return "Koneksi internetku sedang bermasalah atau DuckDuckGo membatasi pencarianku sementara. Coba beberapa saat lagi ya!"
