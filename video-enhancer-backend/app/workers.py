import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import utils


async def process_job(job_id: str, store: Dict[str, dict]) -> None:
    job = store.get(job_id)
    if not job:
        return

    src = job.get("uploadPath")
    if not src or not os.path.exists(src):
        job["status"] = "failed"
        job["progress"] = 0
        return

    # Begin processing
    job["status"] = "processing"
    model = job.get("model", "realesrgan-x4plus")
    # Derive ESRGAN scale from model name
    scale = 4
    if "x4plus" in model or "face" in model or "anime" in model:
        scale = 4
    elif "x3" in model:
        scale = 3
    elif "x2" in model:
        scale = 2
    job["message"] = f"Preparing… (model: {model})"
    job["progress"] = 5

    # Prepare final output path using requested timestamped naming
    orig_name = Path(src).name
    out_abs, public_url = utils.result_enhanced_timestamped_path(orig_name)

    # Working directory for intermediate artifacts
    work_dir = Path(utils.RESULTS_DIR) / f"{job_id}_work"
    frames_dir = work_dir / "frames"
    enhanced_dir = work_dir / "enhanced"
    faces_dir = work_dir / "enhanced_faces"
    audio_path = work_dir / "audio.m4a"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)

        # Ensure ffmpeg available (local auto-download on Windows)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, utils.ensure_ffmpeg)

        # 1) Extract frames from ORIGINAL video for best quality (not from a downscaled intermediate)
        if job.get("canceled"):
            job["status"] = "canceled"; job["message"] = "Canceled by user"; return
        job["message"] = "Extracting frames…"; job["progress"] = 20
        frames_pattern = await loop.run_in_executor(None, utils.extract_frames, str(src), str(frames_dir), 30)

        # 3) Enhance frames via Real-ESRGAN (ncnn-vulkan) with graceful fallback
        if job.get("canceled"):
            job["status"] = "canceled"; job["message"] = "Canceled by user"; return
        job["message"] = f"Preparing Real-ESRGAN… (model: {model})"; job["progress"] = 45
        await loop.run_in_executor(None, utils.ensure_realesrgan)

        try:
            job["message"] = f"Enhancing frames… (model: {model}, scale: {scale}x)"; job["progress"] = 50
            enhanced_dir.mkdir(parents=True, exist_ok=True)
            frames = sorted(frames_dir.glob("frames_*.png"))
            # Pre-check frames exist and are readable (non-zero)
            if not frames:
                raise RuntimeError("No frames extracted")
            # Drop zero-sized frames if any (rare)
            frames = [p for p in frames if p.exists() and p.stat().st_size > 0]
            if not frames:
                raise RuntimeError("No valid frames to enhance")
            total = len(frames)
            job["totalFrames"] = total
            job["processedFrames"] = 0
            log_path = str(work_dir / "logs" / "frame_enhance.log")

            # Parallel enhancement with retries and GPU round-robin
            max_workers = min(max(2, (os.cpu_count() or 4) // 2), 8)
            gpu_pool = [0, 1]  # try balancing if multiple GPUs exist; utils will fallback if not

            def enhance_one(idx_path: Tuple[int, Path]) -> int:
                i, in_path = idx_path
                out_path = enhanced_dir / f"enhanced_{i:06d}.png"
                # Quick input validation
                if not utils.is_image_nonblack(str(in_path)):
                    utils.write_log(log_path, f"input_black frame={i} path={in_path}")
                # Up to 3 attempts (2 retries)
                attempts = 0
                last_err: Optional[Exception] = None
                while attempts < 3:
                    try:
                        gpu_choice = gpu_pool[attempts % len(gpu_pool)] if gpu_pool else None
                        utils.run_realesrgan_frame(str(in_path), str(out_path), model=model, scale=scale, gpu_id=str(gpu_choice) if gpu_choice is not None else None)
                        # Validate output not black
                        if not utils.is_image_nonblack(str(out_path)):
                            raise RuntimeError("output_black")
                        return i
                    except Exception as e:
                        last_err = e
                        attempts += 1
                # Final fallback: copy input frame (may be corrected in post-fix)
                utils.write_log(log_path, f"enhance_fail_final frame={i} err={last_err}")
                utils.copy_frame(str(in_path), str(out_path))
                return i

            indexed = [(idx + 1, p) for idx, p in enumerate(frames)]
            done_count = 0
            start_ts = time.time()
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(enhance_one, ip) for ip in indexed]
                for fut in as_completed(futures):
                    if job.get("canceled"):
                        job["status"] = "canceled"; job["message"] = "Canceled by user"; return
                    _ = fut.result()  # raise if any thread error surfaced
                    done_count += 1
                    job["processedFrames"] = done_count
                    # Update average ms per frame for ETA
                    elapsed = max(0.001, time.time() - start_ts)
                    job["avgMsPerFrame"] = int((elapsed * 1000.0) / max(1, done_count))
                    # Map progress 50% -> 80% based on frames completed
                    pct = 50 + int(30 * (done_count / total))
                    job["progress"] = min(80, pct)
                    job["message"] = f"Enhancing with AI… {done_count}/{total}"

            # Post-fix pass to guarantee non-black and ordered outputs
            last_good: Optional[Path] = None
            for i in range(1, total + 1):
                out_path = enhanced_dir / f"enhanced_{i:06d}.png"
                in_path = frames_dir / f"frames_{i:06d}.png"
                if not out_path.exists() or not utils.is_image_nonblack(str(out_path)):
                    if last_good and last_good.exists():
                        utils.copy_frame(str(last_good), str(out_path))
                    else:
                        # Use input as last resort
                        utils.copy_frame(str(in_path), str(out_path))
                if utils.is_image_nonblack(str(out_path)):
                    last_good = out_path

            # 4) Optional face enhancement via GFPGAN over ESRGAN frames
            use_faces = False
            try:
                job["message"] = "Enhancing faces (GFPGAN)…"; job["progress"] = 72
                await loop.run_in_executor(None, utils.ensure_gfpgan)
                if os.path.exists(utils.GFPGAN_EXE):
                    faces_dir.mkdir(parents=True, exist_ok=True)
                    esr_frames = sorted(enhanced_dir.glob("enhanced_*.png"))
                    total_f = max(1, len(esr_frames))

                    def run_face(idx_path):
                        idx, path = idx_path
                        out_path = faces_dir / f"faces_{idx:06d}.png"
                        utils.run_gfpgan_frame(str(path), str(out_path))
                        return idx

                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [pool.submit(run_face, (i+1, p)) for i, p in enumerate(esr_frames)]
                        donef = 0
                        for fut in as_completed(futures):
                            if job.get("canceled"):
                                job["status"] = "canceled"; job["message"] = "Canceled by user"; return
                            fut.result()
                            donef += 1
                            pct = 72 + int(6 * (donef / total_f))
                            job["progress"] = min(78, pct)
                            job["message"] = f"Enhancing faces (GFPGAN)… {donef}/{total_f}"
                    use_faces = True
            except Exception:
                # Skip GFPGAN on any error
                use_faces = False

            # 5) Extract audio from ORIGINAL (handle errors locally; proceed video-only if needed)
            if job.get("canceled"):
                job["status"] = "canceled"; job["message"] = "Canceled by user"; return
            job["message"] = "Extracting audio…"; job["progress"] = 70
            audio_ok = True
            try:
                await loop.run_in_executor(None, utils.extract_audio, str(src), str(audio_path))
            except Exception as ae:
                audio_ok = False
                job["message"] = f"Audio unavailable: {ae}. Continuing without audio"

            # 6) Combine enhanced frames (or face-enhanced) + audio (or silent)
            if job.get("canceled"):
                job["status"] = "canceled"; job["message"] = "Canceled by user"; return
            job["message"] = "Encoding final 4K video…"; job["progress"] = 92
            pattern_dir = faces_dir if use_faces else enhanced_dir
            name_prefix = "faces" if use_faces else "enhanced"
            enhanced_pattern = str(pattern_dir / f"{name_prefix}_%06d.png")
            try:
                if audio_ok:
                    await loop.run_in_executor(None, utils.combine_frames_audio_2160p, enhanced_pattern, str(audio_path), out_abs, 30)
                else:
                    await loop.run_in_executor(None, utils.combine_frames_video_only_2160p, enhanced_pattern, out_abs, 30)
            except Exception as ce:
                # If combining with audio fails, retry video-only once
                await loop.run_in_executor(None, utils.combine_frames_video_only_2160p, enhanced_pattern, out_abs, 30)
        except Exception as e:
            # Graceful fallback: include precise ESRGAN error in job message for diagnosis
            detail = str(e)
            job["message"] = f"Real-ESRGAN failed: {detail}. Falling back to 4K upscale"
            job["progress"] = 95
            await loop.run_in_executor(None, utils.run_ffmpeg_scale_2160p, src, out_abs)

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Failed: {e}"
        job["progress"] = 100
        return
    finally:
        # Cleanup work dir (best-effort)
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

    # Success
    job["status"] = "completed"
    job["message"] = "Completed"
    job["progress"] = 100
    job["resultPath"] = out_abs
    job["resultUrl"] = public_url
