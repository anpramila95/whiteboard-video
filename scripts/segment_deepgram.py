import json
import re
from pathlib import Path
from typing import Any

DEEPGRAM_FILE = Path("deepgram.json")
OUTPUT_PLAN = Path("storyboard_plan.json")

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def split_deepgram_balanced(target_sec: float = 30.0, max_sec: float = 38.0) -> list[dict[str, Any]]:
    data = json.loads(DEEPGRAM_FILE.read_text(encoding="utf-8"))
    words = data["results"]["channels"][0]["alternatives"][0]["words"]

    # Nhóm theo các đoạn ngắt tự nhiên: khoảng nghỉ giữa 2 từ >= 0.35s
    cues = []
    curr = []
    for i, w in enumerate(words):
        curr.append(w)
        is_last = (i == len(words) - 1)
        next_gap = 0.0 if is_last else (words[i+1]["start"] - w["end"])
        
        # Ngắt câu nếu nghỉ dài hoặc kết thúc
        if next_gap >= 0.35 or is_last:
            txt = " ".join([x.get("punctuated_word", x.get("word", "")) for x in curr])
            cues.append({
                "start": curr[0]["start"],
                "end": curr[-1]["end"],
                "text": txt,
                "words": curr
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
                "text": clean_text(" ".join([c["text"] for c in bucket])),
                "cues": bucket
            })
            bucket = []

    if bucket:
        if scenes and (bucket[-1]["end"] - bucket[0]["start"]) < 15.0:
            scenes[-1]["cues"].extend(bucket)
            scenes[-1]["end"] = bucket[-1]["end"]
            scenes[-1]["duration_ms"] = int((scenes[-1]["end"] - scenes[-1]["start"]) * 1000)
            scenes[-1]["text"] = clean_text(scenes[-1]["text"] + " " + " ".join([c["text"] for c in bucket]))
        else:
            scenes.append({
                "scene_idx": len(scenes) + 1,
                "start": bucket[0]["start"],
                "end": bucket[-1]["end"],
                "duration_ms": int((bucket[-1]["end"] - bucket[0]["start"]) * 1000),
                "text": clean_text(" ".join([c["text"] for c in bucket])),
                "cues": bucket
            })

    return scenes

if __name__ == "__main__":
    scenes = split_deepgram_balanced()
    print(f"Tổng số cảnh: {len(scenes)}")
    max_d = max(s['duration_ms']/1000 for s in scenes)
    min_d = min(s['duration_ms']/1000 for s in scenes)
    print(f"Thời lượng mỗi cảnh: ngắn nhất {min_d:.1f}s, dài nhất {max_d:.1f}s")
