import argparse
import os
import cv2
import numpy as np
from PIL import Image

from model import load_model
from network import (
    get_cassava_area_mask,
    get_deeplab_transform,
    overlay_predictions_on_image,
    predict_with_deeplab,
)


def process_image(image_path, deeplab_model, device, transform, output_dir, use_tta):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Cannot read image: {image_path}")
        return
    cassava_mask = get_cassava_area_mask(img_bgr)
    masked_bgr = img_bgr.copy()
    masked_bgr[cassava_mask == 0] = [0, 0, 0]
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    masked_pil = Image.fromarray(masked_rgb)
    pred_mask = predict_with_deeplab(
        deeplab_model, masked_pil, device, transform, use_tta=use_tta
    )
    overlay = overlay_predictions_on_image(masked_pil, pred_mask)
    cassava_pixels = int(np.count_nonzero(cassava_mask))
    pred_rotten_in_cassava = pred_mask * cassava_mask
    pred_rotten_pixels = int(np.count_nonzero(pred_rotten_in_cassava))
    ppd = pred_rotten_pixels / cassava_pixels if cassava_pixels > 0 else 0.0
    ppd_str = f"{ppd:.4f}"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}_PPD-{ppd_str}_overlay.png")
    Image.fromarray(overlay).save(save_path)
    print(
        f"Saved prediction: {save_path} "
        f"(PPD={ppd_str}, CassavaPixels={cassava_pixels}, RottenPixels={pred_rotten_pixels})"
    )


def collect_image_files(input_path):
    if os.path.isdir(input_path):
        image_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, f))
        return image_files
    if os.path.isfile(input_path):
        return [input_path]
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="./test",
        help="Input image or directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="predict_results",
        help="Output directory",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=os.path.join("models", "best_model.pth"),
        help="Path to model weights",
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--encoder-name", default="resnet101")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--encoder-output-stride", type=int, default=16)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    deeplab_model, device = load_model(
        args.weights,
        num_classes=args.num_classes,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        encoder_output_stride=args.encoder_output_stride,
    )
    transform = get_deeplab_transform()
    image_files = collect_image_files(args.input)
    if not image_files:
        print(f"Invalid input path or no image files: {args.input}")
        return
    print(f"Found {len(image_files)} image files")
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i + 1}/{len(image_files)}: {image_path}")
        try:
            process_image(
                image_path,
                deeplab_model,
                device,
                transform,
                args.output,
                use_tta=not args.no_tta,
            )
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")


if __name__ == "__main__":
    main()
