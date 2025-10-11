import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple
from datetime import datetime
import subprocess
import shutil as _shutil

# Resolve project root from this file: backend_root = .../video-enhancer-backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = BACKEND_ROOT / "uploads"
RESULTS_DIR = BACKEND_ROOT / "results"
FFMPEG_DIR = BACKEND_ROOT / "ffmpeg"
FFMPEG_EXE = FFMPEG_DIR / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

# Real-ESRGAN (ncnn-vulkan portable build for Windows). We prefer a local copy.
REALESRGAN_DIR = BACKEND_ROOT / "realesrgan"
REALESRGAN_EXE = REALESRGAN_DIR / ("realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan")
# Support nested layout from the 20220424 zip without moving files
REALESRGAN_NESTED_DIR = REALESRGAN_DIR / "realesrgan-ncnn-vulkan-20220424-windows"
REALESRGAN_NESTED_EXE = REALESRGAN_NESTED_DIR / ("realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan")
REALESRGAN_MODELS_DIR = REALESRGAN_NESTED_DIR / "models"
# Model name for general-purpose 4x: realesr-general-x4v3 (ncnn uses .param/.bin files)
REALESRGAN_MODEL_NAME = "realesr-general-x4v3"

# Optional GFPGAN (ncnn-vulkan) portable build support
GFPGAN_DIR = BACKEND_ROOT / "gfpgan"
GFPGAN_EXE = GFPGAN_DIR / ("gfpgan-ncnn-vulkan.exe" if os.name == "nt" else "gfpgan-ncnn-vulkan")
GFPGAN_MODEL_NAME = "GFPGANv1.4"


def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    REALESRGAN_DIR.mkdir(parents=True, exist_ok=True)
    GFPGAN_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    keep = (" ", ".", "-", "_")
    cleaned = ''.join(ch for ch in name if ch.isalnum() or ch in keep).strip()
    return cleaned or "file"


def upload_path(job_id: str, filename: str) -> str:
    """Absolute path for uploaded file for a job."""
    base = f"{job_id}__{safe_filename(filename)}"
    return str(UPLOADS_DIR / base)


def result_path(job_id: str, filename: str) -> Tuple[str, str]:
    """
    Returns (abs_path, public_url) for a result file.
    We expose results under /results via StaticFiles (mounted in main.py).
    """
    name = f"{job_id}__enhanced_{safe_filename(filename)}"
    abs_path = RESULTS_DIR / name
    public_url = f"/results/{name}"
    return str(abs_path), public_url


def result_enhanced_timestamped_path(original_filename: str) -> Tuple[str, str]:
    """
    Returns (abs_path, public_url) using the required naming format:
    enhanced_[originalfilename]_[timestamp].mp4
    The original filename is sanitized; timestamp format: YYYYMMDD_HHMMSS
    """
    stem = Path(safe_filename(original_filename)).stem or "video"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"enhanced_{stem}_{ts}.mp4"
    abs_path = RESULTS_DIR / name
    public_url = f"/results/{name}"
    return str(abs_path), public_url


def result_enhanced_path(job_id: str) -> Tuple[str, str]:
    """
    Returns (abs_path, public_url) for an enhanced result file using the
    required naming: results/{jobId}_enhanced.mp4
    """
    name = f"{job_id}_enhanced.mp4"
    abs_path = RESULTS_DIR / name
    public_url = f"/results/{name}"
    return str(abs_path), public_url


def ffmpeg_bin() -> str:
    """Return the absolute path to ffmpeg binary, preferring local copy."""
    if FFMPEG_EXE.exists():
        return str(FFMPEG_EXE)
    # fallback to system ffmpeg
    return "ffmpeg"


def ensure_ffmpeg() -> None:
    """Ensure ffmpeg exists locally. On Windows, download a static build if missing.

    Download source: Gyan.dev release 'ffmpeg-release-essentials.zip'.
    """
    ensure_dirs()
    if FFMPEG_EXE.exists():
        return
    if os.name != "nt":
        # On non-Windows, we expect ffmpeg in PATH; leave as-is
        return
    # Windows download
    zip_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = FFMPEG_DIR / "ffmpeg.zip"
    try:
        urllib.request.urlretrieve(zip_url, str(zip_path))
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(FFMPEG_DIR))
        # Find ffmpeg.exe within extracted directories
        exe_path = None
        for root, dirs, files in os.walk(FFMPEG_DIR):
            if "ffmpeg.exe" in files:
                exe_path = Path(root) / "ffmpeg.exe"
                break
        if not exe_path:
            raise FileNotFoundError("ffmpeg.exe not found after extraction")
        # Move to FFMPEG_DIR/ffmpeg.exe
        shutil.copyfile(str(exe_path), str(FFMPEG_EXE))
    finally:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except Exception:
                pass


def ensure_gfpgan() -> None:
    """Ensure GFPGAN (ncnn-vulkan) exists locally. We only validate presence; no auto-download."""
    ensure_dirs()
    if GFPGAN_EXE.exists():
        return
    # Try to locate if user manually extracted under BACKEND_ROOT
    exe_path = None
    for root, dirs, files in os.walk(BACKEND_ROOT):
        if "gfpgan-ncnn-vulkan.exe" in files:
            exe_path = Path(root) / "gfpgan-ncnn-vulkan.exe"
            break
        if "gfpgan-ncnn-vulkan" in files:
            exe_path = Path(root) / "gfpgan-ncnn-vulkan"
            break
    if exe_path:
        shutil.copyfile(str(exe_path), str(GFPGAN_EXE))
        if os.name != "nt":
            os.chmod(str(GFPGAN_EXE), 0o755)
        return
    # If not found, we leave it unavailable; worker will skip gracefully
    return


def gfpgan_bin() -> str:
    if GFPGAN_EXE.exists():
        return str(GFPGAN_EXE)
    return "gfpgan-ncnn-vulkan"


def find_realesrgan_models_dir() -> str:
    """Return a models directory for Real-ESRGAN if present.
    Priority: nested models dir -> REALESRGAN_DIR/models -> first 'models' found under REALESRGAN_DIR.
    """
    if REALESRGAN_MODELS_DIR.exists() and REALESRGAN_MODELS_DIR.is_dir():
        return str(REALESRGAN_MODELS_DIR)
    direct = REALESRGAN_DIR / "models"
    if direct.exists() and direct.is_dir():
        return str(direct)
    for root, dirs, files in os.walk(REALESRGAN_DIR):
        if "models" in dirs:
            return str(Path(root) / "models")
    return ""


def model_files_available(model_name: str) -> bool:
    """Check if <model>.param and <model>.bin exist under the detected models dir."""
    md = find_realesrgan_models_dir()
    if not md:
        return False
    p = Path(md) / f"{model_name}.param"
    b = Path(md) / f"{model_name}.bin"
    return p.exists() and b.exists()


def resolve_model_param(model_or_param: str) -> str:
    """Resolve a model selection to a full .param path.
    Accepts base name (e.g., 'realesrgan-x4plus') or filename (e.g., 'realesrgan-x4plus.param').
    Returns absolute path to .param; raises if not found.
    """
    md = find_realesrgan_models_dir()
    if not md:
        raise FileNotFoundError("Real-ESRGAN models directory not found")
    # If already an absolute/relative path provided
    candidate = Path(model_or_param)
    if candidate.suffix == ".param":
        # Absolute or relative to models dir
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        in_models = Path(md) / candidate.name
        if in_models.exists():
            return str(in_models.resolve())
        raise FileNotFoundError(f"Model .param not found: {candidate}")
    # Treat as base name
    base = Path(md) / f"{model_or_param}.param"
    if base.exists():
        return str(base.resolve())
    raise FileNotFoundError(f"Model .param not found for base name: {model_or_param}")


def list_available_models() -> list:
    """Return a list of model base names available in the detected models directory."""
    md = find_realesrgan_models_dir()
    if not md:
        return []
    names = set()
    for entry in os.listdir(md):
        if entry.endswith('.param'):
            names.add(entry[:-6])
        elif entry.endswith('.bin'):
            names.add(entry[:-4])
    return sorted(names)


def map_model_base(model: str) -> str:
    """Map UI model selection/value to the actual ncnn base model name.
    General -> realesrgan-x4plus
    Face -> realesrgan-x4plus-anime
    Anime -> realesrgan-x4plus-anime
    AnimeVideoV3 x2/x3/x4 -> realesr-animevideov3-<scale>
    Fallback: return stem/base name.
    """
    # Most UIs pass the value already; still normalize per requested mapping
    m = (model or "").strip().lower()
    mapping = {
        "general": "realesrgan-x4plus",
        "face": "realesrgan-x4plus-anime",
        "anime": "realesrgan-x4plus-anime",
        "realesrgan-x4plus-face": "realesrgan-x4plus-anime",
        "realesrgan-x4plus": "realesrgan-x4plus",
        "realesrgan-x4plus-anime": "realesrgan-x4plus-anime",
        "realesr-animevideov3-x2": "realesr-animevideov3-x2",
        "realesr-animevideov3-x3": "realesr-animevideov3-x3",
        "realesr-animevideov3-x4": "realesr-animevideov3-x4",
    }
    # If a filename is passed, reduce to stem first
    stem = Path(m).stem
    return mapping.get(m, mapping.get(stem, stem))


def ensure_realesrgan() -> None:
    """Ensure Real-ESRGAN (ncnn-vulkan) exists locally with the general x4v3 model.

    On Windows, download a prebuilt archive and extract the exe and model files.
    On non-Windows, we expect the binary to be installed manually or available in PATH.
    """
    ensure_dirs()
    if REALESRGAN_EXE.exists():
        return
    if os.name != "nt":
        # Non-Windows: rely on system installation or manual placement under REALESRGAN_DIR
        return

    # If the user manually extracted the archive elsewhere under BACKEND_ROOT,
    # try to locate and move the files into REALESRGAN_DIR before downloading.
    exe_path = None
    model_param = None
    model_bin = None
    for root, dirs, files in os.walk(BACKEND_ROOT):
        if not exe_path and "realesrgan-ncnn-vulkan.exe" in files:
            exe_path = Path(root) / "realesrgan-ncnn-vulkan.exe"
        if not model_param and f"{REALESRGAN_MODEL_NAME}.param" in files:
            model_param = Path(root) / f"{REALESRGAN_MODEL_NAME}.param"
        if not model_bin and f"{REALESRGAN_MODEL_NAME}.bin" in files:
            model_bin = Path(root) / f"{REALESRGAN_MODEL_NAME}.bin"
        if exe_path and model_param and model_bin:
            break
    if exe_path:
        shutil.copyfile(str(exe_path), str(REALESRGAN_EXE))
        if model_param:
            shutil.copyfile(str(model_param), str(REALESRGAN_DIR / model_param.name))
        if model_bin:
            shutil.copyfile(str(model_bin), str(REALESRGAN_DIR / model_bin.name))
        return
    # Download ncnn-vulkan Windows build (includes models). Try several mirrors/versions.
    candidate_urls = [
        # 20220424 release
        "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.3.0/Real-ESRGAN-ncnn-vulkan-20220424-windows.zip",
        # 20220728 release (alternative tag naming in some forks)
        "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/20220728/Real-ESRGAN-ncnn-vulkan-20220728-windows.zip",
        # 20210210 legacy (as a fallback)
        "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/20210210/Real-ESRGAN-ncnn-vulkan-20210210-windows.zip",
    ]
    zip_path = REALESRGAN_DIR / "realesrgan.zip"
    last_err = None
    try:
        # Use a browser-like user-agent to avoid 403 on some endpoints
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)

        for url in candidate_urls:
            try:
                urllib.request.urlretrieve(url, str(zip_path))
                break
            except Exception as e:
                last_err = e
                continue
        else:
            raise last_err or RuntimeError("Failed to download Real-ESRGAN archive from all mirrors")
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(REALESRGAN_DIR))
        # Find executable and desired model files
        exe_path = None
        model_param = None
        model_bin = None
        for root, dirs, files in os.walk(REALESRGAN_DIR):
            if not exe_path and "realesrgan-ncnn-vulkan.exe" in files:
                exe_path = Path(root) / "realesrgan-ncnn-vulkan.exe"
            if not model_param and f"{REALESRGAN_MODEL_NAME}.param" in files:
                model_param = Path(root) / f"{REALESRGAN_MODEL_NAME}.param"
            if not model_bin and f"{REALESRGAN_MODEL_NAME}.bin" in files:
                model_bin = Path(root) / f"{REALESRGAN_MODEL_NAME}.bin"
        if not exe_path:
            raise FileNotFoundError("realesrgan executable not found after extraction")
        # Copy exe to root dir
        if os.name == "nt":
            shutil.copyfile(str(exe_path), str(REALESRGAN_EXE))
        else:
            shutil.copyfile(str(exe_path), str(REALESRGAN_EXE))
            os.chmod(str(REALESRGAN_EXE), 0o755)
        # Copy model files (if present); optional if exe bundles models
        if model_param:
            shutil.copyfile(str(model_param), str(REALESRGAN_DIR / model_param.name))
        if model_bin:
            shutil.copyfile(str(model_bin), str(REALESRGAN_DIR / model_bin.name))
    finally:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except Exception:
                pass


def realesrgan_bin() -> str:
    # Prefer nested exe if present (no need to move files)
    if REALESRGAN_NESTED_EXE.exists():
        return str(REALESRGAN_NESTED_EXE)
    if REALESRGAN_EXE.exists():
        return str(REALESRGAN_EXE)
    return "realesrgan-ncnn-vulkan"


def run_realesrgan_frame(input_path: str, output_path: str, model: str = REALESRGAN_MODEL_NAME, scale: int = 4, gpu_id: str | None = None) -> None:
    """Enhance a single frame image using Real-ESRGAN (ncnn-vulkan).

    Correct CLI usage:
    -m <models_dir> and -n <model_base_name>, not a .param path.
    Uses absolute paths for input/output, and prefers GPU order 1 -> 0 -> auto.
    Raises RuntimeError with a concise stderr snippet on failure so callers can surface it.
    """
    exe = realesrgan_bin()
    models_dir = find_realesrgan_models_dir()
    if not models_dir:
        raise RuntimeError("Real-ESRGAN models directory not found")
    # Normalize to base model name
    model_base = map_model_base(model)
    # Absolute IO paths
    in_abs = str(Path(input_path).resolve())
    out_abs = str(Path(output_path).resolve())
    base = [
        exe,
        "-m", models_dir,
        "-i", in_abs,
        "-o", out_abs,
        "-n", model_base,
        "-s", str(scale),
        "-f", "png",
    ]
    tried = []
    def with_gpu(args, gflag):
        return [args[0], "-g", gflag] + args[1:]
    variants = []
    if gpu_id is not None:
        variants.append(with_gpu(base, str(gpu_id)))
        # Fall back to the other common ids and auto
        if str(gpu_id) != "0":
            variants.append(with_gpu(base, "0"))
        if str(gpu_id) != "1":
            variants.append(with_gpu(base, "1"))
        variants.append(base)
    else:
        variants = [with_gpu(base, "1"), with_gpu(base, "0"), base]
    last_err = None
    for cmd in variants:
        try:
            # Print the command for diagnostics
            print("[ESRGAN] exec:", " ".join(cmd))
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="ignore") if e.stderr else ""
            tried.append((cmd, err))
            last_err = e
        except Exception as e:
            last_err = e
    # Compose concise error snippet for surfacing in job message
    tail = (tried[-1][1] if tried else "").strip()
    tail_snippet = (tail[:200] + ("…" if len(tail) > 200 else "")) or "<no stderr>"
    raise RuntimeError(f"ESRGAN error: {tail_snippet}") from last_err


# --- Frame validation & utilities ---
def is_image_nonblack(path: str) -> bool:
    """Return True if the image has any non-zero pixel (i.e., not fully black).

    Prefers Pillow for accuracy; falls back to a size heuristic if Pillow isn't available.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        from PIL import Image
        with Image.open(str(p)) as img:
            # Convert to L (grayscale) quickly and check extrema
            g = img.convert("L")
            lo, hi = g.getextrema()
            return (hi or 0) > 0
    except Exception:
        # Fallback: consider non-trivially small PNGs as likely non-black
        # (still better than nothing if PIL isn't installed)
        return p.stat().st_size > 1000


def copy_frame(src: str, dst: str) -> None:
    """Copy frame file from src to dst (overwrite if exists)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    _shutil.copyfile(src, dst)


def write_log(log_path: str, message: str) -> None:
    """Append a message line to the given log file."""
    try:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        pass


def run_gfpgan_frame(input_path: str, output_path: str, model: str = GFPGAN_MODEL_NAME) -> None:
    """Enhance faces on a single frame via GFPGAN (ncnn-vulkan)."""
    base = [gfpgan_bin(), "-i", input_path, "-o", output_path, "-n", model, "-f", "png"]
    tried = []
    def with_gpu(args, gflag):
        return [args[0], "-g", gflag] + args[1:]
    variants = [with_gpu(base, "1"), with_gpu(base, "0"), base, with_gpu(base, "-1")]
    last_err = None
    for cmd in variants:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(GFPGAN_DIR))
            return
        except subprocess.CalledProcessError as e:
            tried.append((cmd, e.stderr.decode(errors="ignore") if e.stderr else ""))
            last_err = e
        except Exception as e:
            last_err = e
    diag = "\n\n".join(["CMD: " + " ".join(c) + "\nERR: " + (s or "<no stderr>") for c, s in tried])
    raise RuntimeError(f"GFPGAN failed for frame. Attempts:\n{diag}") from last_err


def run_ffmpeg_scale_720p(input_path: str, output_path: str) -> None:
    """Resize video to 1280x720 and copy audio if possible."""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i", input_path,
        "-vf", "scale=1280:720",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def combine_frames_audio_2160p(frames_pattern: str, audio_path: str, output_path: str, fps: int = 30) -> None:
    """Combine frames and audio into final MP4, scaling to 2160p height (4K) with lanczos."""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-i", audio_path,
        "-vf", "scale=-2:2160:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def combine_frames_video_only_2160p(frames_pattern: str, output_path: str, fps: int = 30) -> None:
    """Combine frames into a silent MP4, scaling to 2160p height (4K) with lanczos.

    Used when audio extraction fails or the source has no audio stream.
    """
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-vf", "scale=-2:2160:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "18",
        "-an",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_ffmpeg_scale_2160p(input_path: str, output_path: str) -> None:
    """Scale a video to 2160p height (4K), preserve aspect ratio (width multiple of 2)."""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i", input_path,
        "-vf", "scale=-2:2160:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def combine_frames_audio_1080p(frames_pattern: str, audio_path: str, output_path: str, fps: int = 30) -> None:
    """Combine frames and audio into final MP4, scaling to 1080p height with lanczos."""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-i", audio_path,
        "-vf", "scale=-2:1080:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_ffmpeg_scale_1080p(input_path: str, output_path: str) -> None:
    """Scale a video to 1080p height, preserve aspect ratio (width multiple of 2)."""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i", input_path,
        "-vf", "scale=-2:1080:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_frames(input_path: str, frames_dir: str, fps: int = 30) -> str:
    """Extract frames to frames_dir as frames_%06d.png. Returns pattern path."""
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    pattern = str(Path(frames_dir) / "frames_%06d.png")
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i", input_path,
        "-vf", f"fps={fps}",
        pattern,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return pattern


def extract_audio(input_path: str, audio_out: str) -> None:
    """Extract and (re)encode audio to AAC to ensure compatibility."""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i", input_path,
        "-vn",
        "-c:a", "aac",
        "-b:a", "192k",
        audio_out,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore") if e.stderr else ""
        # Common case: source has no audio stream; let caller decide to proceed without audio
        raise RuntimeError((err[:240] + ("…" if len(err) > 240 else "")) or "ffmpeg extract_audio failed") from e


def combine_frames_audio(frames_pattern: str, audio_path: str, output_path: str, fps: int = 30) -> None:
    """Combine frames and audio into final MP4."""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
