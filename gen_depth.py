import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from moge.model.v2 import MoGeModel

device = torch.device("cuda")


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """将 depth 归一化到 0–255，大于 255 的置 0，返回 uint8。"""
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
    # 按你的要求：超过 255 的设为 0（归一化后理论上不会，此处作保险）
    out[out > 255] = 0
    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="视频逐帧导出 RGB + Depth")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", default="./output", help="输出根目录，其下创建 rgb 与 depth")
    parser.add_argument("--model", default="Ruicheng/moge-2-vitl-normal", help="MoGe 预训练模型名或路径")
    args = parser.parse_args()

    rgb_dir = Path(args.output) / "rgb"
    depth_dir = Path(args.output) / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    model = MoGeModel.from_pretrained(args.model).to(device).eval()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {args.input}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = total if total > 0 else None

    frame_id = 0
    with tqdm(total=total, desc="处理视频", unit="frame") as pbar:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # 与 123.py / infer 一致：(3, H, W), [0, 1], float32
            x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

            with torch.inference_mode():
                out = model.infer(x)
            depth = out["depth"].cpu().numpy().squeeze()

            depth_u8 = normalize_depth_to_uint8(depth)

            frame_id += 1
            rgb_name = f"{frame_id:05d}.jpg"
            depth_name = f"{frame_id:05d}.png"
            cv2.imwrite(str(rgb_dir / rgb_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(depth_dir / depth_name), depth_u8)

            pbar.update(1)

    cap.release()
    print(f"共处理 {frame_id} 帧 -> {rgb_dir} & {depth_dir}")


if __name__ == "__main__":
    main()