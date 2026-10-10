import argparse
import os
import struct
import numpy as np
from PIL import Image

def export_mnist_samples(image_path, label_path, output_dir="demo/samples", count=15):
    # Check current path and project root relative path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    if not os.path.exists(image_path):
        candidate = os.path.join(project_root, image_path)
        if os.path.exists(candidate):
            image_path = candidate

    if not os.path.exists(label_path):
        candidate = os.path.join(project_root, label_path)
        if os.path.exists(candidate):
            label_path = candidate

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print(f"[!] Error: Dataset files not found -> '{image_path}' or '{label_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(image_path, 'rb') as f_img, open(label_path, 'rb') as f_lbl:
        # Read IDX headers (Big-Endian format)
        magic_img, total_imgs, rows, cols = struct.unpack(">IIII", f_img.read(16))
        magic_lbl, total_lbls = struct.unpack(">II", f_lbl.read(8))

        export_count = min(count, total_imgs)
        print(f"[*] Found {total_imgs} samples. Exporting first {export_count} images to '{output_dir}'...")

        for idx in range(export_count):
            raw_pixels = f_img.read(rows * cols)
            label = struct.unpack(">B", f_lbl.read(1))[0]

            img_arr = np.frombuffer(raw_pixels, dtype=np.uint8).reshape((rows, cols))
            img = Image.fromarray(img_arr)

            file_name = f"sample_{idx}_label_{label}.png"
            file_path = os.path.join(output_dir, file_name)
            img.save(file_path)

        print(f"[+] Successfully exported {export_count} images to '{output_dir}/'.")

def main():
    parser = argparse.ArgumentParser(description="Export MNIST binary dataset samples as PNG images")
    parser.add_argument("--images", type=str, default="data/t10k-images.idx3-ubyte", help="Path to MNIST images IDX file")
    parser.add_argument("--labels", type=str, default="data/t10k-labels.idx1-ubyte", help="Path to MNIST labels IDX file")
    parser.add_argument("--output", type=str, default="demo/samples", help="Output directory for PNG images (default: demo/samples)")
    parser.add_argument("--count", type=int, default=15, help="Number of samples to export (default: 15)")
    args = parser.parse_args()

    export_mnist_samples(args.images, args.labels, args.output, args.count)

if __name__ == "__main__":
    main()
