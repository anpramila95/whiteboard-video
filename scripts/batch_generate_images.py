import json
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

API_URL = "http://localhost:3001/v1/images/generations"
STYLE_PREFIX = (
    "Whiteboard animation line art, clean continuous black outlines on plain warm cream paper background (#F5EBD7), "
    "vector illustration style, minimalist coloring, distinct separated subjects across the scene, wide 16:9 shot, no text: "
)

def build_prompt_for_scene(scene: dict[str, Any]) -> str:
    # Sinh prompt mô tả hình vẽ whiteboard dựa theo nội dung cảnh
    text = scene["text"].lower()
    idx = scene["scene_idx"]
    
    # Prompt trực quan hóa nội dung
    if idx == 1:
        return "Cozy bedroom at night, a person relaxing peacefully on a comfortable bed, large open window showing starry night sky outside"
    elif idx == 2:
        return "Cosmic origin concept, glowing distant galaxies swirling, a warm cup of tea and a blanket on a desk beside an open window"
    elif idx == 3:
        return "Scientific exploration, a giant observatory telescope pointing towards outer space, alongside a magnifying glass looking into tiny fundamental particles"
    elif "nguyên tử" in text or "hạt nhân" in text or "electron" in text:
        return "Atomic structure diagram, tiny nucleus in the center surrounded by orbiting electron cloud rings, stadium scale comparison"
    elif "ngôi sao" in text or "bụi sao" in text:
        return "Birth and lifecycle of stars, giant glowing nebula collapsing and exploding into cosmic star dust, releasing heavy elements"
    elif "kỷ nguyên tối" in text or "sương mù" in text:
        return "Cosmic dark ages, massive ocean of cold hydrogen and helium gas fog floating silently in deep space before first stars ignite"
    elif "anten" in text or "arnold" in text or "bức xạ" in text:
        return "Two scientists standing beside a giant horn antenna telescope, looking up at the sky, detecting cosmic microwave background radiation"
    elif "ti vi" in text or "đốm tuyết" in text:
        return "Vintage CRT television set displaying black and white static noise fuzz on screen, in a cozy living room"
    elif "lò phản ứng" in text or "big bang" in text or "proton" in text:
        return "The primordial furnace minutes after Big Bang, colliding protons and neutrons fusing into early hydrogen and helium nuclei"
    elif "quark" in text or "gluon" in text or "lhc" in text or "gia tốc" in text:
        return "Large Hadron Collider underground circular tunnel and particle detector, colliding heavy lead ions creating droplet of primordial quark-gluon plasma"
    elif "phản vật chất" in text or "positron" in text:
        return "Cosmic battle of matter and antimatter, pairs of opposite particles colliding, annihilating into flashes of pure light, slight asymmetry"
    elif "vật chất tối" in text or "hố đen" in text:
        return "Spiral galaxy rotating with invisible dark matter web scaffolding holding stars together, mysterious primordial black holes"
    elif "bàn tay" in text or "ngụm nước" in text or "trái đất" in text:
        return "Human hand reaching forward, connection between ancient water atoms, primordial hydrogen from Big Bang, and living human body"
    else:
        # Tự động tóm tắt ý chính cho các cảnh còn lại
        words = scene["text"].split()[:20]
        snippet = " ".join(words)
        return f"Whiteboard storytelling scene illustrating: {snippet}"

def generate_and_save_image(scene: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    idx = scene["scene_idx"]
    out_file = output_dir / f"scene-{idx:02d}.png"
    if out_file.exists() and out_file.stat().st_size > 1000:
        print(f"[bỏ qua] Cảnh {idx:02d} đã có ảnh: {out_file.name}")
        scene["image_path"] = str(out_file)
        return scene

    prompt = build_prompt_for_scene(scene)
    scene["prompt"] = prompt
    full_prompt = STYLE_PREFIX + prompt

    payload = json.dumps({"prompt": full_prompt, "ratio": "16:9"}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        img_url = data.get("url")
        if not img_url and "data" in data and len(data["data"]) > 0:
            img_url = data["data"][0].get("url")

        if not img_url:
            raise ValueError(f"Không nhận được URL từ API: {data}")

        # Tải ảnh về
        urllib.request.urlretrieve(img_url, str(out_file))
        print(f"[ok] Cảnh {idx:02d} tạo thành công -> {out_file.name}")
        scene["image_path"] = str(out_file)
    except Exception as e:
        print(f"[lỗi] Cảnh {idx:02d} thất bại: {e}")
        scene["image_path"] = None

    return scene

def run_batch_generation(scenes: list[dict[str, Any]], output_dir: Path, max_workers: int = 6):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Bắt đầu tạo {len(scenes)} ảnh song song ({max_workers} luồng)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_and_save_image, sc, output_dir) for sc in scenes]
        for f in as_completed(futures):
            f.result()

if __name__ == "__main__":
    from segment_deepgram import split_deepgram_balanced
    scenes = split_deepgram_balanced()
    
    # Lưu storyboard kế hoạch
    plan_path = Path("storyboard_plan.json")
    for s in scenes:
        s["prompt"] = build_prompt_for_scene(s)
    plan_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lập kế hoạch {len(scenes)} cảnh -> {plan_path.name}")
