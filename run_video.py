import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Import renderer, tts và config
ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import tts_narration as tts  # type: ignore[import-not-found]
from render_stream_whiteboard import RegionStreamRenderer  # type: ignore[import-not-found]
from stream_render import (  # type: ignore[import-not-found]
    DEFAULT_HAND_PNG,
    Config,
    _imread_any,
    transcode_h264,
)

# Cấu hình API LLM & Sinh Ảnh
LLM_API_URL = "http://localhost:20128/v1/chat/completions"
LLM_MODEL = "ag/gemini-3.8-flash"
IMAGE_API_URL = "http://localhost:3001/v1/images/generations"

STYLE_PREFIX = (
    "Whiteboard animation line art, clean continuous black outlines on plain warm cream paper background (#F5EBD7), "
    "vector illustration style, minimalist coloring, distinct separated subjects across the scene, wide 16:9 shot, no text: "
)

# ──────────────────────────────────────────────────────────────
# TÙY CHỌN PHONG CÁCH VẼ & TÔ MÀU
# ──────────────────────────────────────────────────────────────
# "brush"  : (Mặc định như cũ) Đi nét phác thảo trước, sau đó đầu bút tô màu nhẹ nhàng theo nét vẽ
# "direct" : Vẽ trực tiếp ra màu chuẩn ngay khi ngòi bút đi qua (nhanh, màu sắc nét tức thì)
# "wipe"   : Đi nét trước, sau đó quét lớp màu từ trên xuống dưới
DEFAULT_COLOR_MODE = "brush"


def load_env() -> dict[str, str]:
    env_vars = {}
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
    return env_vars


ENV = load_env()
OPENAI_API_KEY = ENV.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


def split_deepgram(dg_path: Path, target_sec: float = 30.0, max_sec: float = 38.0) -> list[dict[str, Any]]:
    data = json.loads(dg_path.read_text(encoding="utf-8"))
    words = data["results"]["channels"][0]["alternatives"][0]["words"]

    cues = []
    curr = []
    for i, w in enumerate(words):
        curr.append(w)
        is_last = (i == len(words) - 1)
        next_gap = 0.0 if is_last else (words[i+1]["start"] - w["end"])
        if next_gap >= 0.35 or is_last:
            txt = " ".join([x.get("punctuated_word", x.get("word", "")) for x in curr])
            cues.append({
                "start": curr[0]["start"],
                "end": curr[-1]["end"],
                "text": txt,
            })
            curr = []

    scenes = []
    bucket = []
    for cue in cues:
        bucket.append(cue)
        dur = bucket[-1]["end"] - bucket[0]["start"]
        if dur >= target_sec or dur >= max_sec:
            scenes.append({
                "scene_idx": len(scenes) + 1,
                "start": bucket[0]["start"],
                "end": bucket[-1]["end"],
                "duration_ms": int((bucket[-1]["end"] - bucket[0]["start"]) * 1000),
                "text": re.sub(r"\s+", " ", " ".join([c["text"] for c in bucket])).strip(),
                "cues": bucket,
            })
            bucket = []

    if bucket:
        if scenes and (bucket[-1]["end"] - bucket[0]["start"]) < 15.0:
            scenes[-1]["cues"].extend(bucket)
            scenes[-1]["end"] = bucket[-1]["end"]
            scenes[-1]["duration_ms"] = int((scenes[-1]["end"] - scenes[-1]["start"]) * 1000)
            scenes[-1]["text"] = re.sub(r"\s+", " ", scenes[-1]["text"] + " " + " ".join([c["text"] for c in bucket])).strip()
        else:
            scenes.append({
                "scene_idx": len(scenes) + 1,
                "start": bucket[0]["start"],
                "end": bucket[-1]["end"],
                "duration_ms": int((bucket[-1]["end"] - bucket[0]["start"]) * 1000),
                "text": re.sub(r"\s+", " ", " ".join([c["text"] for c in bucket])).strip(),
                "cues": bucket,
            })
    return scenes


# ── ĐỌC VÀ KHỚP CẢNH TỪ FILE TEXT.TXT (ĐỊNH DẠNG: text | prompt) ──
def load_scenes_from_text_file(text_path: Path, dg_path: Path) -> list[dict[str, Any]]:
    print(f"\n=== ĐỌC KỊCH BẢN VÀ PROMPT TỪ {text_path.name} ===")
    lines = text_path.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Chỉ coi là comment nếu bắt đầu bằng # mà KHÔNG có dấu ngăn cách |
        if line.startswith("#") and "|" not in line:
            continue
        if "|" in line:
            txt, prompt = line.split("|", 1)
            entries.append((txt.strip(), prompt.strip()))
        else:
            entries.append((line.strip(), ""))

    if not entries:
        raise ValueError(f"File {text_path.name} không có nội dung hợp lệ.")

    print(f"Tổng số dòng kịch bản trong {text_path.name}: {len(entries)} (tương ứng {len(entries)} cảnh)")

    if not dg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file deepgram để khớp mốc thời gian: {dg_path}")

    print(f"Đang tham chiếu mốc thời gian từ: {dg_path}...")
    data = json.loads(dg_path.read_text(encoding="utf-8"))
    dg_words = data["results"]["channels"][0]["alternatives"][0]["words"]

    # Xây dựng luồng ký tự liên tục từ Deepgram
    char_to_word = []
    full_chars = []
    for w_idx, w in enumerate(dg_words):
        clean = re.sub(r"[\s\W]", "", w.get("punctuated_word", w.get("word", "")).lower())
        for c in clean:
            full_chars.append(c)
            char_to_word.append(w_idx)

    full_stream = "".join(full_chars)
    total_chars = len(full_stream)

    def clean_txt(t: str) -> str:
        # Lọc bỏ tiêu đề markdown (##, ###), số thứ tự (1., 2.), ký tự đặc biệt
        t = re.sub(r"^[\s#\d\.\-①②③④⑤⑥⑦⑧⑨⑩]+", "", t)
        return re.sub(r"[\s\W#*①②③④⑤⑥⑦⑧⑨⑩]", "", t.lower())

    # Dò tìm mốc bắt đầu của từng dòng theo n-gram anchor
    cursor = 0
    scene_positions = []
    for idx, (scene_text, prompt) in enumerate(entries):
        cl = clean_txt(scene_text)
        found_pos = -1
        # Tìm cụm 5-6 ký tự đầu tiên khớp trong luồng âm thanh
        for k in range(0, min(max(0, len(cl) - 5), 35), 2):
            chunk = cl[k:k+6]
            pos = full_stream.find(chunk, cursor)
            if pos != -1 and pos < cursor + 350:
                found_pos = pos
                break
        if found_pos == -1:
            found_pos = cursor

        scene_positions.append((found_pos, len(cl), scene_text, prompt))
        cursor = found_pos + int(len(cl) * 0.7)

    # Đảm bảo 100% số dòng trong text.txt đều trở thành cảnh (không bao giờ bị bỏ sót)
    scenes = []
    for i in range(len(scene_positions)):
        start_pos = scene_positions[i][0]
        scene_len = scene_positions[i][1]
        scene_text = scene_positions[i][2]
        prompt = scene_positions[i][3]

        if i < len(scene_positions) - 1:
            end_pos = scene_positions[i + 1][0]
        else:
            end_pos = total_chars - 1

        if end_pos <= start_pos:
            end_pos = min(total_chars - 1, start_pos + scene_len)

        w_start = char_to_word[min(start_pos, len(char_to_word) - 1)]
        w_end = char_to_word[min(end_pos, len(char_to_word) - 1)]
        scene_start = dg_words[w_start]["start"]
        scene_end = dg_words[w_end]["end"]
        dur_ms = max(1000, int((scene_end - scene_start) * 1000))

        matched = dg_words[w_start : w_end + 1]
        if not matched:
            matched = [dg_words[w_start]]

        # Chia nhỏ thành các sub-cues để lộ hình dần theo câu
        sub_cues = []
        chunk = []
        for w in matched:
            chunk.append(w)
            pw = w.get("punctuated_word", w.get("word", ""))
            if pw.endswith((".", "?", "!", ",", ";", "、", "。")) or len(chunk) >= 12:
                sub_cues.append({
                    "start": chunk[0]["start"],
                    "end": chunk[-1]["end"],
                    "text": " ".join([x.get("punctuated_word", x.get("word", "")) for x in chunk]),
                })
                chunk = []
        if chunk:
            sub_cues.append({
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
                "text": " ".join([x.get("punctuated_word", x.get("word", "")) for x in chunk]),
            })

        idx = i + 1
        scenes.append({
            "scene_idx": idx,
            "start": scene_start,
            "end": scene_end,
            "duration_ms": dur_ms,
            "text": scene_text,
            "prompt": prompt,
            "cues": sub_cues,
        })
        print(f"  + Cảnh {idx:02d} ({scene_start:.2f}s -> {scene_end:.2f}s, {dur_ms/1000:.1f}s): {prompt[:50]}...")

    return scenes


# ── BƯỚC 1: DÙNG LLM SINH PROMPT & LƯU CACHE (CHẠY TIẾP TỤC) ──
def generate_all_prompts(scenes: list[dict[str, Any]], work_dir: Path, force: bool = False) -> None:
    prompts_cache_file = work_dir / "prompts_cache.json"
    cached_prompts: dict[str, str] = {}

    if not force and prompts_cache_file.exists():
        try:
            cached_prompts = json.loads(prompts_cache_file.read_text(encoding="utf-8"))
        except Exception:
            cached_prompts = {}

    # Lọc những cảnh chưa có prompt
    needed_scenes = []
    for sc in scenes:
        if sc.get("prompt"):
            cached_prompts[str(sc["scene_idx"])] = sc["prompt"]
            continue
        idx_str = str(sc["scene_idx"])
        if idx_str in cached_prompts and not force:
            sc["prompt"] = cached_prompts[idx_str]
        else:
            needed_scenes.append(sc)

    if not needed_scenes:
        print(f"[sẵn sàng] Đã có đủ {len(scenes)} prompt (từ file text / cache, không cần gọi LLM)")
        prompts_cache_file.write_text(json.dumps(cached_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    print(f"\n=== BƯỚC 1: DÙNG LLM ({LLM_MODEL}) SINH PROMPT CHO {len(needed_scenes)}/{len(scenes)} CẢNH CHƯA CÓ ===")
    scenes_input = [{"scene_idx": sc["scene_idx"], "text": sc["text"]} for sc in needed_scenes]

    system_prompt = (
        "You are an expert whiteboard animation director. "
        "You will be given a list of consecutive narration scenes from a video. "
        "For EACH scene, write a clear, concise visual scene prompt in English for image generation.\n"
        "Guidelines:\n"
        "1. Write in English.\n"
        "2. Focus on clear visual subjects (characters, objects, actions) placed distinctly across the 16:9 canvas.\n"
        "3. Keep each prompt under 35 words. No meta-commentary, no styling words like 'whiteboard' or 'cream background'.\n"
        "4. Output MUST be a valid JSON array of objects, each with 'scene_idx' (integer) and 'prompt' (string).\n"
        "Example format:\n"
        "[{\"scene_idx\": 1, \"prompt\": \"A person looking out through a large window...\"}, {\"scene_idx\": 2, \"prompt\": \"...\"}]"
    )
    user_prompt = f"Here is the list of scenes:\n{json.dumps(scenes_input, ensure_ascii=False, indent=2)}\n\nRespond ONLY with the raw JSON array."

    payload = json.dumps({
        "model": LLM_MODEL,
        "stream": False,
        "temperature": 0.5,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    req = urllib.request.Request(LLM_API_URL, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        prompt_list = json.loads(content)
        prompt_map = {item["scene_idx"]: item["prompt"] for item in prompt_list if "scene_idx" in item and "prompt" in item}

        for sc in needed_scenes:
            idx = sc["scene_idx"]
            if idx in prompt_map:
                sc["prompt"] = prompt_map[idx]
            else:
                words = sc["text"].split()[:18]
                sc["prompt"] = f"Illustration showing: {' '.join(words)}"
            cached_prompts[str(idx)] = sc["prompt"]
            print(f"[Prompt Cảnh {idx:02d}]: {sc['prompt']}")

        # Lưu lại cache file
        prompts_cache_file.write_text(json.dumps(cached_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] Đã lưu cache prompt vào {prompts_cache_file.name}")

    except Exception as e:
        print(f"[Cảnh báo] Lỗi gọi LLM ({e}). Fallback trích xuất từ câu thoại.")
        for sc in needed_scenes:
            words = sc["text"].split()[:18]
            sc["prompt"] = f"Illustration showing: {' '.join(words)}"
            cached_prompts[str(sc["scene_idx"])] = sc["prompt"]
        prompts_cache_file.write_text(json.dumps(cached_prompts, ensure_ascii=False, indent=2), encoding="utf-8")


# ── BƯỚC 2: TẠO ẢNH SONG SONG DỰA VÀO PROMPT ĐÃ SINH ──
def generate_single_image(scene: dict[str, Any], out_path: Path, force: bool = False, max_retries: int = 3) -> Path | None:
    if not force and out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[đã có] Cảnh {scene['scene_idx']:02d}: {out_path.name}")
        return out_path

    prompt = scene.get("prompt")
    if not prompt:
        words = scene["text"].split()[:18]
        prompt = f"Illustration showing: {' '.join(words)}"
    full_prompt = STYLE_PREFIX + prompt
    payload = json.dumps({"prompt": full_prompt, "ratio": "16:9"}).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(IMAGE_API_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            img_url = data.get("url") or (data.get("data", [{}])[0].get("url") if data.get("data") else None)
            if not img_url:
                raise ValueError(f"Không có URL trong response: {data}")
            urllib.request.urlretrieve(img_url, str(out_path))
            print(f"[ok] Đã tạo ảnh Cảnh {scene['scene_idx']:02d}: {out_path.name}")
            return out_path
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(f"[lỗi] Cảnh {scene['scene_idx']:02d} tạo ảnh thất bại sau {max_retries} lần thử: {e}")
                return None


def generate_all_images(scenes: list[dict[str, Any]], work_dir: Path, workers: int = 6, force: bool = False) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== BƯỚC 2: TẠO {len(scenes)} ẢNH TỪ PROMPT ({workers} LUỒNG) ===")

    img_paths = []
    scenes_to_generate = []

    for sc in scenes:
        idx = sc["scene_idx"]
        # Kiểm tra xem ảnh đã có sẵn chưa: ưu tiên 1.png, 01.png, scene_01.png
        candidates = [
            work_dir / f"{idx}.png",
            work_dir / f"{idx:02d}.png",
            work_dir / f"scene_{idx:02d}.png",
            work_dir / f"scene_{idx}.png",
            work_dir / f"{idx}.jpg",
            work_dir / f"{idx}.jpeg",
        ]
        found = None
        if not force:
            for c in candidates:
                if c.exists() and c.stat().st_size > 1000:
                    found = c
                    break

        if found:
            print(f"[đã có] Cảnh {idx:02d} dùng lại ảnh có sẵn: {found.name}")
            img_paths.append(found)
        else:
            # Tạo mới và lưu theo dạng 1.png, 2.png,...
            target_path = work_dir / f"{idx}.png"
            img_paths.append(target_path)
            scenes_to_generate.append((sc, target_path))

    if scenes_to_generate:
        print(f"Cần tạo mới {len(scenes_to_generate)}/{len(scenes)} ảnh...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(generate_single_image, sc, path, force): sc for sc, path in scenes_to_generate}
            for f in as_completed(futures):
                f.result()
    else:
        print(f"[sẵn sàng] Tất cả {len(scenes)} ảnh đã có sẵn trong folder, không cần tạo mới!")

    return img_paths


# ── BƯỚC CẮT AUDIO TỪ FILE MASTER_TTS.WAV THEO TIMELINE DEEPGRAM ──
def extract_audio_from_master(scenes: list[dict[str, Any]], master_audio: Path, work_dir: Path) -> Path:
    print(f"\n=== CẮT AUDIO TỪ {master_audio.name} THEO TIMELINE DEEPGRAM ===")
    total_start = scenes[0]["start"]
    total_end = scenes[-1]["end"]
    total_dur = total_end - total_start

    audio_track = work_dir / "full_narration.wav"
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    # Dùng ffmpeg cắt đúng đoạn của các cảnh này
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", f"{total_start:.3f}",
        "-t", f"{total_dur:.3f}",
        "-i", str(master_audio),
        "-c:a", "copy",
        str(audio_track)
    ]
    subprocess.run(cmd, check=True)
    print(f"[ok] Đã trích xuất audio đoạn {total_start:.2f}s -> {total_end:.2f}s ({total_dur:.2f}s) vào {audio_track.name}")
    return audio_track


# ── BƯỚC TẠO ANNOTATION KHỚP VỚI TIMELINE DEEPGRAM GỐC ──
def auto_detect_regions(img_bgr: np.ndarray, cues: list[dict[str, Any]], scene_start: float, total_ms: int) -> dict[str, Any]:
    h, w = img_bgr.shape[:2]
    elements = []

    # Chọn số vùng phù hợp (3 vùng nếu cảnh dài >= 20s, 2 vùng nếu ngắn)
    scene_dur_s = total_ms / 1000.0
    n_splits = 3 if scene_dur_s >= 20.0 else 2
    col_w = w // n_splits

    # Phân bổ đều các cues theo dòng thời gian để các vùng trải đều toàn bộ thời lượng cảnh
    target_bucket_dur = scene_dur_s / n_splits
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_splits)]
    for c in cues:
        b_idx = int((c["start"] - scene_start) / target_bucket_dur)
        b_idx = min(n_splits - 1, max(0, b_idx))
        buckets[b_idx].append(c)

    cur_timeline_ms = 0
    for i in range(n_splits):
        b = buckets[i]
        subtitle = " ".join([c["text"] for c in b]) if b else f"Phần {i+1}"

        if b:
            b_start_ms = max(0, int((b[0]["start"] - scene_start) * 1000))
            b_end_ms = min(total_ms, int((b[-1]["end"] - scene_start) * 1000))
        else:
            b_start_ms = cur_timeline_ms
            b_end_ms = int(cur_timeline_ms + target_bucket_dur * 1000)

        # Đảm bảo các vùng nối tiếp nhau liên tục bám sát giọng đọc
        start_ms = max(cur_timeline_ms, b_start_ms)
        dur_ms = max(1000, b_end_ms - start_ms)
        cur_timeline_ms = start_ms + dur_ms

        elements.append({
            "id": f"elem_{i+1}",
            "label": f"Vùng {i+1}",
            "sequence": i + 1,
            "subtitle": subtitle,
            "region": {
                "x": i * col_w,
                "y": 0,
                "width": col_w if i < n_splits - 1 else (w - i * col_w),
                "height": h,
            },
            "reveal": {
                "direction": "top_to_bottom",
                "startMs": start_ms,
                "durationMs": dur_ms,
                "maskPaddingPx": 0,
                "protectedRegions": [],
            },
        })

    return {
        "canvas": {"width": w, "height": h},
        "sceneDurationMs": total_ms,
        "elements": elements,
    }


# ── BƯỚC 4: RENDER TỪNG CẢNH ──
def render_scene_video(img_path: Path, scene: dict[str, Any], output_mp4: Path, color_mode: str = "brush") -> Path:
    img_bgr = _imread_any(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")

    ann_path = img_path.with_suffix(".annotation.json")
    ann_data = auto_detect_regions(img_bgr, scene["cues"], scene["start"], scene["duration_ms"])
    ann_path.write_text(json.dumps(ann_data, ensure_ascii=False, indent=2), encoding="utf-8")

    if color_mode == "direct":
        # Vẽ trực tiếp màu gốc theo đầu bút, không cần bước tô màu riêng
        cfg = Config(
            fps=30,
            cap_long_edge=1080,
            gaze_seconds=1.0,
            ink_path_mode="skeleton",
            sample_step=3,
            ink_weight=1,
            color_weight=0,
        )
    elif color_mode == "wipe":
        # Đi nét trước, sau đó quét lớp màu từ trên xuống dưới
        cfg = Config(
            fps=30,
            cap_long_edge=1080,
            gaze_seconds=1.0,
            ink_path_mode="skeleton",
            sample_step=3,
            ink_weight=2,
            color_weight=1,
            color_fill="contour-wipe",
        )
    else:  # "brush" (mặc định như cũ)
        # Đi nét phác thảo trước, sau đó đầu bút tô màu nhẹ nhàng theo nét vẽ
        cfg = Config(
            fps=30,
            cap_long_edge=1080,
            gaze_seconds=1.0,
            ink_path_mode="skeleton",
            sample_step=3,
            ink_weight=2,
            color_weight=1,
            color_fill="brush",
            brush_radius=35,
        )

    renderer = RegionStreamRenderer(
        image_bgr=img_bgr,
        annotation=ann_data,
        cfg=cfg,
        hand_png=DEFAULT_HAND_PNG,
        bare_tip=False,
    )
    
    raw_mp4 = output_mp4.with_name(f"{output_mp4.stem}_raw.mp4")
    renderer.render_to(raw_mp4, total_ms=scene["duration_ms"])
    final = transcode_h264(raw_mp4, output_mp4)
    raw_mp4.unlink(missing_ok=True)
    return final


# ── BƯỚC 5: GHÉP VIDEO ──
def concat_videos(video_list: list[Path], output_path: Path) -> Path:
    print(f"\n=== BƯỚC 5: GHÉP {len(video_list)} CẢNH THÀNH VIDEO HOÀN CHỈNH ===")
    list_txt = output_path.with_name("concat_list.txt")
    with open(list_txt, "w", encoding="utf-8") as f:
        for vp in video_list:
            f.write(f"file '{vp.resolve().as_posix()}'\n")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_txt), "-c", "copy", str(output_path)
        ], check=True)
    else:
        from merge_scenes import _pyav_concat  # type: ignore[import-not-found]
        _pyav_concat(video_list, output_path)

    list_txt.unlink(missing_ok=True)
    return output_path


# ── BƯỚC TẠO GIỌNG ĐỌC & GHÉP VÀO VIDEO ──
def generate_and_attach_audio(scenes: list[dict[str, Any]], video_path: Path, work_dir: Path) -> Path:
    print(f"\n=== BƯỚC GHÉP GIỌNG ĐỌC TTS (EDGE-TTS) ===")
    tts_cache_dir = work_dir / "tts-cache"
    tts_cache_dir.mkdir(parents=True, exist_ok=True)

    voice = ENV.get("TTS_VOICE") or "vi-VN-NamMinhNeural"
    speed = float(ENV.get("TTS_SPEED") or 1.0)
    audio_clips: list[Path] = []
    cues_list: list[dict[str, Any]] = []

    # Gom các câu thoại để đọc
    for sc in scenes:
        for cue in sc["cues"]:
            cues_list.append({
                "index": len(cues_list) + 1,
                "text": cue["text"],
                "startMs": int(cue["start"] * 1000),
                "endMs": int(cue["end"] * 1000),
            })

    print(f"Tổng số {len(cues_list)} câu thoại cần sinh giọng Edge-TTS ({voice})...")

    def _tts_single(cue: dict[str, Any]) -> tuple[int, Path]:
        idx = cue["index"]
        text = cue["text"]
        out_clip = tts_cache_dir / f"cue_{idx:03d}.mp3"
        if not out_clip.exists() or out_clip.stat().st_size < 100:
            tts.synthesize_edge(text, voice, speed, out_clip)
        # Cắt khoảng lặng thừa
        trim_clip = tts.trim_silence(out_clip)
        return idx, trim_clip

    clips_map = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(_tts_single, c) for c in cues_list]
        for f in as_completed(futures):
            idx, p = f.result()
            clips_map[idx] = p

    audio_clips = [clips_map[c["index"]] for c in cues_list]

    # Xây dựng track âm thanh hoàn chỉnh
    audio_track = work_dir / "full_narration.m4a"
    total_ms = scenes[-1]["duration_ms"] + int(scenes[-1]["start"] * 1000)
    tts.build_track(cues_list, audio_clips, audio_track, total_ms=total_ms)

    # Ghép âm thanh vào video MP4 bằng FFmpeg
    video_with_audio = video_path.with_name(f"{video_path.stem}_voice.mp4")
    tts.mux(video_path, audio_track, video_with_audio)
    print(f"[ok] Đã ghép thành công giọng đọc vào video: {video_with_audio.name}")
    return video_with_audio


def main():
    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("  1. Truyền folder dự án: python run_video.py <đường_dẫn_folder> [limit_scenes] [--force]")
        print("  2. Truyền file text:    python run_video.py text.txt [limit_scenes] [--force]")
        print("  3. Truyền deepgram:     python run_video.py deepgram.json [limit_scenes] [--force]")
        sys.exit(1)

    raw_input = Path(sys.argv[1]).resolve()
    if not raw_input.exists():
        print(f"Lỗi: Không tìm thấy {raw_input}")
        sys.exit(1)

    force = "--force" in sys.argv
    limit = None
    for arg in sys.argv[2:]:
        if arg.isdigit():
            limit = int(arg)
            break

    # Nếu truyền vào là thư mục -> mọi thao tác và file xuất ra nằm ngay trong thư mục đó
    if raw_input.is_dir():
        project_dir = raw_input
        text_candidates = [
            project_dir / "text.txt",
            project_dir / "script.txt",
        ]
        text_file = next((f for f in text_candidates if f.exists()), None)
    else:
        project_dir = raw_input.parent
        text_file = raw_input if raw_input.suffix.lower() == ".txt" else None

    # Tìm deepgram.json
    dg_candidates = [
        project_dir / ".make_video_v2" / "deepgram.json",
        project_dir / "deepgram.json",
        ROOT_DIR / ".make_video_v2" / "deepgram.json",
        ROOT_DIR / "deepgram.json",
    ]
    dg_path = next((f for f in dg_candidates if f.exists()), None)

    # Tìm master_tts.wav / mp3
    audio_candidates = [
        project_dir / "master_tts.wav",
        project_dir / "master_tts.mp3",
        project_dir / ".make_video_v2" / "master_tts.wav",
        project_dir / ".make_video_v2" / "master_tts.mp3",
        ROOT_DIR / "master_tts.wav",
    ]
    master_audio = next((f for f in audio_candidates if f.exists()), None)

    if text_file:
        if not dg_path:
            print("Lỗi: Đã có text.txt nhưng không tìm thấy deepgram.json (trong .make_video_v2/ hoặc cùng thư mục) để khớp mốc thời gian!")
            sys.exit(1)
        scenes = load_scenes_from_text_file(text_file, dg_path)
    elif dg_path:
        print("=== PHÂN TÍCH DEEPGRAM & LẬP KẾ HOẠCH CẢNH ===")
        scenes = split_deepgram(dg_path)
    else:
        print(f"Lỗi: Trong folder {project_dir} không có text.txt hoặc deepgram.json!")
        sys.exit(1)

    if limit:
        scenes = scenes[:limit]
        print(f"Chạy giới hạn {limit} cảnh đầu.")

    # Xác định chế độ tô màu: CLI flag -> mặc định DEFAULT_COLOR_MODE ("brush")
    color_mode = DEFAULT_COLOR_MODE
    for arg in sys.argv:
        if arg in ["--direct", "--color-mode=direct"]:
            color_mode = "direct"
        elif arg in ["--wipe", "--color-mode=wipe"]:
            color_mode = "wipe"
        elif arg in ["--brush", "--color-mode=brush"]:
            color_mode = "brush"

    print(f"Tổng số cảnh thực hiện: {len(scenes)} (chế độ tô màu: {color_mode})")

    # 1. Prompt (nếu text.txt đã có prompt thì nạp thẳng, không gọi LLM)
    generate_all_prompts(scenes, project_dir, force=force)

    # 2. Tạo ảnh (ưu tiên kiểm tra 1.png, 2.png,... nếu có rồi thì giữ nguyên)
    img_paths = generate_all_images(scenes, project_dir, workers=6, force=force)

    # 3. Render video từng cảnh song song (mặc định 3 luồng cùng lúc)
    render_workers = 3
    print(f"\n=== BƯỚC 3: RENDER VIDEO SONG SONG (30 FPS, 1080p, Region Mask - {render_workers} LUỒNG, mode={color_mode}) ===")
    rendered_dict: dict[int, Path] = {}
    tasks_to_render = []

    for i, sc in enumerate(scenes):
        idx = sc["scene_idx"]
        img_p = img_paths[i]

        vid_candidates = [
            project_dir / f"{idx}.mp4",
            project_dir / f"scene_{idx:02d}.mp4",
        ]
        existing_vid = next((v for v in vid_candidates if v.exists() and v.stat().st_size > 10000), None)
        vid_p = project_dir / f"{idx}.mp4"

        if not img_p.exists() or img_p.stat().st_size < 1000:
            print(f"[bỏ qua] Cảnh {idx:02d} thiếu file ảnh {img_p.name}, vui lòng tạo lại ảnh.")
            continue

        if not force and existing_vid:
            print(f"[đã có] Bỏ qua cảnh {idx:02d} (đã render: {existing_vid.name})")
            rendered_dict[i] = existing_vid
            continue

        tasks_to_render.append((i, img_p, sc, vid_p))

    def _render_worker(task):
        task_i, task_img_p, task_sc, task_vid_p = task
        print(f"\n[Bắt đầu render] Cảnh {task_sc['scene_idx']:02d}/{len(scenes)} ({task_sc['duration_ms']/1000:.1f}s, mode={color_mode})...")
        out_v = render_scene_video(task_img_p, task_sc, task_vid_p, color_mode=color_mode)
        print(f"\n[Xong] Cảnh {task_sc['scene_idx']:02d}: {task_vid_p.name}")
        return task_i, out_v

    if tasks_to_render:
        print(f"Tiến hành render {len(tasks_to_render)} cảnh với {render_workers} cảnh chạy cùng lúc...")
        with ThreadPoolExecutor(max_workers=render_workers) as ex:
            futures = [ex.submit(_render_worker, t) for t in tasks_to_render]
            for f in as_completed(futures):
                task_i, out_v = f.result()
                rendered_dict[task_i] = out_v

    # Sắp xếp đúng theo thứ tự cảnh 1 -> N
    rendered_videos = [rendered_dict[i] for i in sorted(rendered_dict.keys())]

    # 4. Ghép video thành phẩm ngay trong folder đó
    final_video = project_dir / "final_video.mp4"
    concat_videos(rendered_videos, final_video)

    # 5. Cắt audio từ master_tts.wav ghép vào video
    if master_audio and master_audio.exists():
        audio_clip = extract_audio_from_master(scenes, master_audio, project_dir)
        final_with_voice = project_dir / "final_video_voice.mp4"
        tts.mux(final_video, audio_clip, final_with_voice)
        print(f"\n🎉 HOÀN THÀNH TOÀN BỘ!")
        print(f"Video thành phẩm có âm thanh: {final_with_voice.resolve()}")
    else:
        print(f"\n[Lưu ý] Không tìm thấy file master_tts.wav trong folder, xuất video không tiếng: {final_video.resolve()}")


if __name__ == "__main__":
    main()
