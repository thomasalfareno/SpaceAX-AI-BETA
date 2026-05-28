"""
SpaceaxAI - Configuration
Konfigurasi utama dengan auto-detect spesifikasi hardware.
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import platform
import multiprocessing
import torch

# Optimasi CUDA Tensor Cores & CUDNN
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
from dataclasses import dataclass, field

# Optimasi CPU Threads
try:
    cores = multiprocessing.cpu_count()
    # Sisakan 1 core untuk OS agar tidak hang
    torch.set_num_threads(max(1, cores - 1))
except Exception:
    pass

def get_system_ram_gb() -> float:
    """Deteksi total RAM sistem dalam GB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    return 4.0  # Default konservatif

def get_available_ram_gb() -> float:
    """Deteksi RAM tersedia (belum dipakai) dalam GB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            mem = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mem[parts[0].rstrip(':')] = int(parts[1])
            available = mem.get('MemAvailable', mem.get('MemFree', 0))
            return available / (1024 * 1024)
    except Exception:
        pass
    return 2.0

def get_gpu_vram_gb() -> float:
    """Deteksi total memori VRAM GPU dalam GB."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass
    return 0.0

# ============================================================================
# Profil Model — dari SMALL hingga ULTRA
# ============================================================================

MODEL_PROFILES = {
    "small": {
        "d_model": 384, "n_heads": 6, "n_layers": 6,
        "d_ff": 1536, "max_seq_len": 512, "vocab_size": 64000,
        "batch_size": 16, "label": "SMALL (~25M params)",
        "min_ram_gb": 4.0,
    },
    "medium": {
        "d_model": 768, "n_heads": 12, "n_layers": 12,
        "d_ff": 3072, "max_seq_len": 1024, "vocab_size": 64000,
        "batch_size": 8, "label": "MEDIUM (~100M params)",
        "min_ram_gb": 8.0,
    },
    "large": {
        "d_model": 1024, "n_heads": 16, "n_layers": 16,
        "d_ff": 4096, "max_seq_len": 1024, "vocab_size": 64000,
        "batch_size": 4, "label": "LARGE (~250M params)",
        "min_ram_gb": 16.0,
    },
    "ultra": {
        "d_model": 1280, "n_heads": 20, "n_layers": 24,
        "d_ff": 5120, "max_seq_len": 1024, "vocab_size": 64000,
        "batch_size": 2, "label": "ULTRA (~650M params)",
        "min_ram_gb": 32.0,
    },
    # Alias — sub-tier dipilih di auto_model_config via core.promax
    "promax": {
        "d_model": 1536, "n_heads": 24, "n_layers": 28,
        "d_ff": 6144, "max_seq_len": 1024, "vocab_size": 64000,
        "batch_size": 1, "label": "PROMAX (auto-tier 1B/4B/8B)",
        "min_ram_gb": 48.0,
    },
}

def auto_model_config(size_override: str = None):
    """Pilih konfigurasi model otomatis berdasarkan RAM atau override manual."""
    from core.debug_log import agent_log

    total_ram = get_system_ram_gb()
    avail_ram = get_available_ram_gb()
    vram_gb = get_gpu_vram_gb()

    print(f"🖥️  Hardware terdeteksi:")
    print(f"   Total RAM: {total_ram:.1f} GB")
    print(f"   RAM Tersedia: {avail_ram:.1f} GB")
    print(f"   OS: {platform.system()} {platform.machine()}")
    print(f"   CPU Cores: {multiprocessing.cpu_count()}")
    print(f"   CUDA: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ Tidak tersedia (CPU mode)'}")

    # Pilih profil
    if size_override and size_override in MODEL_PROFILES:
        profile_name = size_override
        print(f"   📌 Profil manual: {size_override.upper()}")
    else:
        # Auto-detect berdasarkan RAM
        if total_ram >= 48.0:
            profile_name = "promax"
        elif total_ram >= 32.0:
            profile_name = "ultra"
        elif total_ram >= 16.0:
            profile_name = "large"
        elif total_ram >= 8.0:
            profile_name = "medium"
        else:
            profile_name = "small"

    promax_tier = None
    if profile_name == "promax":
        from core.promax import resolve_promax_tier, get_promax_profile
        force = os.environ.get("SPACEAX_PROMAX_TIER")
        promax_tier = resolve_promax_tier(total_ram, vram_gb, force_tier=force)
        profile = get_promax_profile(promax_tier)
        # #region agent log
        agent_log(
            "config.py:auto_model_config",
            "promax_tier_resolved",
            {
                "tier": promax_tier,
                "total_ram_gb": round(total_ram, 2),
                "vram_gb": round(vram_gb, 2),
                "est_params": profile.get("est_params"),
                "vocab_size": profile.get("vocab_size"),
                "size_override": size_override,
            },
            hypothesis_id="H1",
        )
        # #endregion
    else:
        profile = MODEL_PROFILES[profile_name]

    # Warn jika RAM mungkin tidak cukup (hanya jika auto-detect, bukan manual override)
    if not size_override and total_ram < profile["min_ram_gb"]:
        print(f"   ⚠️  RAM mungkin tidak cukup untuk profil {profile_name.upper()} "
              f"(butuh {profile['min_ram_gb']}GB, punya {total_ram:.1f}GB)")
        print(f"   ⚠️  Menurunkan ke profil yang lebih kecil...")
        for pname in ["small", "medium", "large", "ultra", "promax"]:
            if total_ram >= MODEL_PROFILES[pname]["min_ram_gb"]:
                profile_name = pname
                profile = MODEL_PROFILES[pname]

    cfg = ModelConfig(
        d_model=profile["d_model"],
        n_heads=profile["n_heads"],
        n_layers=profile["n_layers"],
        d_ff=profile["d_ff"],
        max_seq_len=profile["max_seq_len"],
        vocab_size=profile["vocab_size"],
        use_gradient_checkpointing=profile_name in ["ultra", "promax"],
    )
    batch = profile["batch_size"]
    label = profile["label"]

    print(f"   🧠 Profil Model: {label}")
    if promax_tier:
        print(f"      🏆 ProMax tier: {promax_tier}")
    print(f"      d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"n_layers={cfg.n_layers}, d_ff={cfg.d_ff}")
    print(f"      vocab_size={cfg.vocab_size}, max_seq_len={cfg.max_seq_len}")

    return cfg, batch, label, promax_tier, profile_name


@dataclass
class ModelConfig:
    """Konfigurasi arsitektur model."""
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    vocab_size: int = 64000
    dropout: float = 0.1
    rope_theta: float = 10000.0
    use_gradient_checkpointing: bool = False

@dataclass
class TrainingConfig:
    """Konfigurasi untuk proses training."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Efektif batch size = batch_size * ini
    learning_rate: float = 3e-4
    num_epochs: int = 20  # Lebih banyak epoch untuk dataset besar
    warmup_steps: int = 1000  # Warmup lebih lama untuk stabilitas
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    checkpoint_interval: int = 500
    fp16: bool = True  # Aktifkan mixed precision di GPU
    use_bfloat16_cpu: bool = False
    num_workers: int = 0
    optimizer_type: str = "adamw"  # "adamw" atau "adafactor"
    # Early stopping
    early_stopping_patience: int = 5  # Lebih sabar sebelum berhenti

@dataclass
class PathConfig:
    """Konfigurasi path direktori."""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "data")
    seed_dir: str = os.path.join(data_dir, "seed")
    checkpoints_dir: str = os.path.join(data_dir, "checkpoints")
    knowledge_dir: str = os.path.join(data_dir, "knowledge")
    memories_dir: str = os.path.join(data_dir, "memories")
    vocab_dir: str = os.path.join(data_dir, "vocab")
    personality_dir: str = os.path.join(data_dir, "personality")
    kbbi_dir: str = os.path.join(base_dir, "kbbi")

    def ensure_dirs(self):
        dirs = [self.data_dir, self.seed_dir, self.checkpoints_dir,
                self.knowledge_dir, self.memories_dir, self.vocab_dir,
                self.personality_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

# Identitas AI
AI_IDENTITY = {
    "name": "SpaceAx AI",
    "team": "Space Ax Corp",
    "developer": "Thomas Alfareno Ananta Nugraha",
    "university": "Institut Teknologi Sepuluh Nopember Surabaya",
    "faculty": "Fakultas Teknologi Elektro dan Informatika Cerdas (FTEIC)",
    "department": "Departemen Teknik Informatika",
    "program": "Prodi Teknik Informatika",
    "version": "2.0.0",
}

@dataclass
class EmotionConfig:
    emotions: list[str] = field(default_factory=lambda: [
        "joy", "sadness", "anger", "fear",
        "surprise", "disgust", "trust", "anticipation", "neutral"
    ])
    default_emotion: str = "neutral"
    decay_rate: float = 0.05

def get_config(auto_detect: bool = True, size_override: str = None):
    """Dapatkan semua config sekaligus.
    
    Args:
        auto_detect: Jika True, auto-detect hardware untuk pilih profil model
        size_override: Override manual profil model ('small', 'medium', 'large', 'ultra')
    """
    paths = PathConfig()
    paths.ensure_dirs()

    if auto_detect:
        model_cfg, batch_size, label, promax_tier, profile_name = auto_model_config(
            size_override
        )
        training_cfg = TrainingConfig(batch_size=batch_size)

        if promax_tier or size_override == "promax":
            from core.promax import apply_promax_training_overrides
            apply_promax_training_overrides(training_cfg)
            print(
                f"   🏆 ProMax training: epochs≥{training_cfg.num_epochs}, "
                f"warmup={training_cfg.warmup_steps}, "
                f"accum={training_cfg.gradient_accumulation_steps}"
            )
        
        # CPU Mode Optimization
        if not torch.cuda.is_available():
            print("   ⚠️  CPU Mode terdeteksi: Menurunkan batch size.")
            old_batch = training_cfg.batch_size
            training_cfg.batch_size = min(4, max(1, old_batch // 4))
            factor = max(1, old_batch // training_cfg.batch_size)
            training_cfg.gradient_accumulation_steps = training_cfg.gradient_accumulation_steps * factor
            training_cfg.use_bfloat16_cpu = False
            print(f"      Batch Size disesuaikan: {old_batch} → {training_cfg.batch_size}")
            print(f"      Gradient Accumulation Steps: {training_cfg.gradient_accumulation_steps}")
        else:
            # GPU VRAM Optimization
            vram = get_gpu_vram_gb()
            if vram > 0:
                if vram >= 75.0:
                    multiplier = 8
                elif vram >= 38.0:
                    multiplier = 4
                elif vram >= 20.0:
                    multiplier = 2
                else:
                    multiplier = 1
                
                if multiplier > 1:
                    old_batch = training_cfg.batch_size
                    training_cfg.batch_size = old_batch * multiplier
                    old_accum = training_cfg.gradient_accumulation_steps
                    training_cfg.gradient_accumulation_steps = max(1, old_accum // multiplier)
                    print(f"   ⚡ GPU VRAM Terdeteksi: {vram:.1f} GB. Menaikkan Batch Size: {old_batch} → {training_cfg.batch_size} (Accumulation: {old_accum} → {training_cfg.gradient_accumulation_steps})")
        
        # Tentukan optimizer_type otomatis
        total_ram = get_system_ram_gb()
        is_large_model = size_override in ["ultra", "promax"] or promax_tier or (
            size_override is None and total_ram >= 32.0
        )
        if is_large_model or total_ram < 16.0:
            training_cfg.optimizer_type = "adafactor"
            print(f"   ⚙️  RAM Manajemen: Mengaktifkan optimizer low-memory 'adafactor' (menghemat ~9.6GB RAM).")
        else:
            training_cfg.optimizer_type = "adamw"
    else:
        model_cfg = ModelConfig()
        training_cfg = TrainingConfig()
        profile_name = size_override or "medium"

    return {
        "model": model_cfg,
        "training": training_cfg,
        "paths": paths,
        "emotion": EmotionConfig(),
        "identity": AI_IDENTITY,
        "profile_name": profile_name if auto_detect else "medium",
        "promax_tier": promax_tier if auto_detect else None,
        "is_promax": bool(promax_tier or size_override == "promax"),
    }
