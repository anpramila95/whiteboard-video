import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


IMAGE_API_URL = "http://localhost:3001/v1/images/generations"
STYLE_PROMPT_PREFIX = (
    "Whiteboard animation line art, clean black outlines on warm cream paper background (#F5EBD7), "
    "vector style, minimalist coloring, distinct separated subjects across the scene, no text, no messy shading: "
)


def parse_deepgram_sentences(data: dict[str, Any]) -> list[dict[str, Any]]:
    # Trích xuất danh sách câu từ cấu trúc chuẩn của Deepgram
    results = data.get("results", {})
    channels = results.get("channels", [])
    if not channels:
        raise ValueError("Deepgram JSON thiếu 'results.channels'")

    alt = channels[0].get("alternatives", [{}])[0]
    
    # Ưu tiên lấy sentences nếu deepgram bật tiện ích câu
    paragraphs = alt.get("paragraphs", {}).get("paragraphs", [])
    sentences = []
    if paragraphs:
        for p in paragraphs:
            for s in p.get("sentences", []):
                sentences.append({
                    "text": s.get("text", "").strip(),
                    "start": s.get("start", 0.0),
                    "end": s.get("end", 0.0),
                })

    if not sentences and "words" in alt:
        words = alt["words"]
        curr_words = []
        start_t = words[0]["start"]
        for w in words:
            curr_words.append(w.get("punctuated_word", w.get("word", "")))
            if w.get("word", "").endswith((".", "?", "!")) or w.get("punctuated_word", "").endswith((".", "?", "!")):
                sentences.append({
                    "text": " ".join(curr_words),
                    "start": start_t,
                    "end": w.get("end", 0.0),
                })
                curr_words = []
                start_t = None
            elif start_t is None:
                start_t = w.get("start", 0.0)
        if curr_words:
            sentences.append({
                "text": " ".join(curr_words),
                "start": start_t if start_t is not None else 0.0,
                "end": words[-1].get("end", 0.0),
            })

    return sentences


def group_sentences_into_scenes(sentences: list[dict[str, Any]], target_duration: float = 25.0) -> list[dict[str, Any]]:
    # Gom cụm câu thành từng cảnh khoảng 20-30s
    scenes = []
    curr_cues = []
    for s in sentences:
        curr_cues.append(s)
        dur = curr_cues[-1]["end"] - curr_cues[0]["start"]
        if dur >= target_duration:
            scenes.append({
                "scene_idx": len(scenes) + 1,
                "start": curr_cues[0]["start"],
                "end": curr_cues[-1]["end"],
                "duration_ms": int((curr_cues[-1]["end"] - curr_cues[0]["start"]) * 1000),
                "sentences": curr_cues,
            })
            curr_cues = []

    if curr_cues:
        if scenes and (curr_cues[-1]["end"] - curr_cues[0]["start"]) < 10.0:
            # Gộp vào cảnh trước nếu đoạn cuối quá ngắn
            scenes[-1]["sentences"].extend(curr_cues)
            scenes[-1]["end"] = curr_cues[-1]["end"]
            scenes[-1]["duration_ms"] = int((scenes[-1]["end"] - scenes[-1]["start"]) * 1000)
        else:
            scenes.append({
                "scene_idx": len(scenes) + 1,
                "start": curr_cues[0]["start"],
                "end": curr_cues[-1]["end"],
                "duration_ms": int((curr_cues[-1]["end"] - curr_cues[0]["start"]) * 1000),
                "sentences": curr_cues,
            })
    return scenes


def call_image_generation_api(prompt: str, ratio: str = "16:9") -> str:
    # Gửi POST request tới API local để lấy URL ảnh
    payload = json.dumps({"prompt": STYLE_PROMPT_PREFIX + prompt, "ratio": ratio}).encode("utf-8")
    req = urllib.request.Request(
        IMAGE_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))

    # Hỗ trợ cả 2 dạng: data[0].url hoặc url trực tiếp
    if "url" in res_data:
        return res_data["url"]
    if "data" in res_data and len(res_data["data"]) > 0:
        return res_data["data"][0].get("url", "")
    raise ValueError(f"Không tìm thấy URL ảnh trong response: {res_data}")


def download_image(url: str, output_path: Path) -> Path:
    # Tải ảnh về lưu vào thư mục
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(output_path))
    return output_path
