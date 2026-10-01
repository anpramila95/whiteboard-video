import sys
from pathlib import Path

# Thêm thư mục scripts vào sys.path
SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from stream_render import (  # type: ignore[import-not-found]
    DEFAULT_HAND_PNG,
    Config,
    StreamBoardRenderer,
    _imread_any,
    transcode_h264,
)


def render_image_to_video(
    image_path: str | Path,
    output_path: str | Path = "output.mp4",
    duration_ms: int = 6000,
    fps: int = 30,
    gaze_seconds: float = 1.0,
    ink_weight: int = 2,
    color_weight: int = 1,
) -> Path:
    src_path = Path(image_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {src_path.resolve()}")

    img_bgr = _imread_any(str(src_path))
    if img_bgr is None:
        raise ValueError(f"Không thể giải mã dữ liệu ảnh: {src_path.resolve()}")

    dst = Path(output_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw_tmp = dst.with_name(f"{dst.stem}_raw{dst.suffix}")

    cfg = Config(
        fps=fps,
        cap_long_edge=1080,           # Giữ nguyên Full HD sắc nét
        gaze_seconds=gaze_seconds,
        ink_weight=ink_weight,
        color_weight=color_weight,
        ink_path_mode="skeleton",     # Đi nét theo sợi xương thật: vừa thật hơn, vừa bỏ qua dập lưới
        sample_step=3,                # Tối ưu bước nhảy bút
    )
    renderer = StreamBoardRenderer(
        image_bgr=img_bgr,
        cfg=cfg,
        hand_png=DEFAULT_HAND_PNG,
        bare_tip=False,
    )

    renderer.render_to(raw_tmp, total_ms=duration_ms)
    return transcode_h264(raw_tmp, dst)


if __name__ == "__main__":
    input_img = sys.argv[1] if len(sys.argv) > 1 else "examples/4.png"
    output_vid = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
    duration_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 6000

    # ponytail: CLI tối giản 2 tham số. Thêm argparse khi cần tuỳ biến fps, style tô màu.
    print(f"Bắt đầu render: {input_img} -> {output_vid}")
    res = render_image_to_video(input_img, output_vid, duration_ms=duration_ms, fps=30)
    print(f"Xong: {res}")
