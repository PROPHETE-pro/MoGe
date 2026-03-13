import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import itertools
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict depth for all images in a directory")
    parser.add_argument("--image_dir", required=True, type=str, help="Input image directory")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for predicted depth files")
    parser.add_argument("--pretrained", dest="pretrained_model_name_or_path", default=None, type=str, help="Pretrained model name or local path")
    parser.add_argument("--version", dest="model_version", choices=["v1", "v2"], default="v2", type=str, help='Model version, defaults to "v2"')
    parser.add_argument("--device", dest="device_name", default="cuda", type=str, help='Device name, e.g. "cuda", "cuda:0", "cpu"')
    parser.add_argument("--fp16", dest="use_fp16", action="store_true", help="Use fp16 precision for faster inference")
    parser.add_argument("--resize", dest="resize_to", default=None, type=int, help="Resize image long side before inference")
    parser.add_argument("--resolution_level", default=9, type=int, help="Inference resolution level [0-9], ignored if --num_tokens is set")
    parser.add_argument("--num_tokens", default=None, type=int, help="Number of tokens used for inference")
    parser.add_argument("--save_vis", action="store_true", help="Also save colorized depth visualization png")
    return parser.parse_args()


def main():
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from moge.model import import_model_class_by_version
    from moge.utils.vis import colorize_depth

    args = parse_args()
    input_root = Path(args.image_dir)
    output_root = Path(args.output_dir)
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    include_suffices = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]
    image_paths = sorted(itertools.chain(*(input_root.rglob(f"*.{suffix}") for suffix in include_suffices)))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {input_root}")

    if args.pretrained_model_name_or_path is None:
        default_pretrained_by_version = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        args.pretrained_model_name_or_path = default_pretrained_by_version[args.model_version]

    device = torch.device(args.device_name)
    model = import_model_class_by_version(args.model_version).from_pretrained(args.pretrained_model_name_or_path).to(device).eval()
    if args.use_fp16:
        model.half()

    for image_path in tqdm(image_paths, desc="Predicting depth", disable=len(image_paths) <= 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Skip unreadable file: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if args.resize_to is not None:
            height, width = min(args.resize_to, int(args.resize_to * height / width)), min(args.resize_to, int(args.resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)

        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        output = model.infer(
            image_tensor,
            resolution_level=args.resolution_level,
            num_tokens=args.num_tokens,
            use_fp16=args.use_fp16,
        )
        depth = output["depth"].detach().cpu().numpy().astype(np.float32)

        rel = image_path.relative_to(input_root)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        depth_npy_path = out_dir / f"{rel.stem}_depth.npy"
        np.save(depth_npy_path, depth)

        if args.save_vis:
            depth_vis = colorize_depth(depth)
            depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{rel.stem}_depth_vis.png"), depth_vis_bgr)

    print(f"Done. Predicted depth for {len(image_paths)} image(s). Output dir: {output_root}")


if __name__ == "__main__":
    main()
