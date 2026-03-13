import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from moge.model.v2 import MoGeModel


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """将 depth 归一化到 0-255 并返回 uint8。"""
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.uint8)

    low, high = np.nanquantile(depth[valid], [0.01, 0.99])
    if high <= low:
        out = np.zeros_like(depth, dtype=np.float64)
    else:
        out = (depth - low) / (high - low) * 255.0
    out = np.clip(out, 0, 255).astype(np.float64)
    out[out > 255] = 0
    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="对 image_dir 中每张图预测 depth")
    parser.add_argument("--input", "-i", required=True, help="输入图片目录 image_dir")
    parser.add_argument("--output", "-o", default="./output_depth", help="输出目录")
    parser.add_argument("--model", default="Ruicheng/moge-2-vitl-normal", help="MoGe 预训练模型名或路径")
    parser.add_argument("--device", default="cuda", help='推理设备，如 "cuda"、"cuda:0"、"cpu"')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted([p for p in input_dir.rglob("*") if p.suffix in image_exts])
    if len(image_paths) == 0:
        raise FileNotFoundError(f"在目录中未找到图片: {input_dir}")

    device = torch.device(args.device)
    model = MoGeModel.from_pretrained(args.model).to(device).eval()

    saved = 0
    for image_path in tqdm(image_paths, desc="处理图片", unit="img"):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] 跳过无法读取图片: {image_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

        with torch.inference_mode():
            out = model.infer(x)
        depth = out["depth"].detach().cpu().numpy().squeeze()
        depth_u8 = normalize_depth_to_uint8(depth)

        rel = image_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), depth_u8)
        saved += 1

    print(f"完成: 共输出 {saved} 张 depth 图到 {output_dir}")


if __name__ == "__main__":
    main()