import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import itertools
from typing import Optional

import click


@click.command(help="Predict depth for all images in a directory")
@click.option("--image_dir", type=click.Path(exists=True, file_okay=False), required=True, help="Input image directory")
@click.option("--output_dir", type=click.Path(), required=True, help="Output directory for predicted depth files")
@click.option("--pretrained", "pretrained_model_name_or_path", type=str, default=None, help="Pretrained model name or local path")
@click.option("--version", "model_version", type=click.Choice(["v1", "v2"]), default="v2", help='Model version, defaults to "v2"')
@click.option("--device", "device_name", type=str, default="cuda", help='Device name, e.g. "cuda", "cuda:0", "cpu"')
@click.option("--fp16", "use_fp16", is_flag=True, help="Use fp16 precision for faster inference")
@click.option("--resize", "resize_to", type=int, default=None, help="Resize image long side before inference")
@click.option("--resolution_level", type=int, default=9, help="Inference resolution level [0-9], ignored if --num_tokens is set")
@click.option("--num_tokens", type=int, default=None, help="Number of tokens used for inference")
@click.option("--save_vis", is_flag=True, help="Also save colorized depth visualization png")
def main(
    image_dir: str,
    output_dir: str,
    pretrained_model_name_or_path: Optional[str],
    model_version: str,
    device_name: str,
    use_fp16: bool,
    resize_to: Optional[int],
    resolution_level: int,
    num_tokens: Optional[int],
    save_vis: bool,
):
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from moge.model import import_model_class_by_version
    from moge.utils.vis import colorize_depth

    input_root = Path(image_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    include_suffices = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]
    image_paths = sorted(itertools.chain(*(input_root.rglob(f"*.{suffix}") for suffix in include_suffices)))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {input_root}")

    if pretrained_model_name_or_path is None:
        default_pretrained_by_version = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = default_pretrained_by_version[model_version]

    device = torch.device(device_name)
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    if use_fp16:
        model.half()

    for image_path in tqdm(image_paths, desc="Predicting depth", disable=len(image_paths) <= 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            click.echo(f"[WARN] Skip unreadable file: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)

        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        output = model.infer(
            image_tensor,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16,
        )
        depth = output["depth"].detach().cpu().numpy().astype(np.float32)

        rel = image_path.relative_to(input_root)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        depth_npy_path = out_dir / f"{rel.stem}_depth.npy"
        np.save(depth_npy_path, depth)

        if save_vis:
            depth_vis = colorize_depth(depth)
            depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{rel.stem}_depth_vis.png"), depth_vis_bgr)

    click.echo(f"Done. Predicted depth for {len(image_paths)} image(s). Output dir: {output_root}")


if __name__ == "__main__":
    main()
