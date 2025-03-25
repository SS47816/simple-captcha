from pathlib import Path

import cv2
import numpy as np


def extract_templates(input_folder="./data/input/", label_folder="./data/output/", output_folder="./data/template/", num_chars=5, save_imgs=False):
    input_path = Path(input_folder)
    label_path = Path(label_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define fixed bounding boxes (x, y, width, height)
    bboxes = [(5 + 9*i, 11, 8, 10) for i in range(5)]

    # Dictionary to store accumulated templates and counts
    templates = {}
    template_counts = {}

    # Get all .jpg files in the folder
    image_files = list(input_path.glob("*.jpg"))

    for image_path in image_files:
        new_stem = image_path.stem.replace("input", "output")
        label_file = label_path / (new_stem + ".txt")

        # Read label file
        if label_file.exists():
            with open(label_file, "r") as f:
                labels = f.read().strip()

            if len(labels) != num_chars:
                print(f"Warning: Label file {label_file} does not have exactly {num_chars} characters.")
                continue
        else:
            print(f"Warning: Label file {label_file} not found.")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Process each bounding box and corresponding character label
        for i, (x, y, w, h) in enumerate(bboxes):
            char_crop = thresh[y:y+h, x:x+w].astype(np.float32)
            label = labels[i]

            if label not in templates:
                templates[label] = np.zeros_like(char_crop, dtype=np.float32)
                template_counts[label] = 0

            templates[label] += char_crop
            template_counts[label] += 1

    # Save averaged templates
    for label, accumulated_img in templates.items():
        templates[label] = (accumulated_img / template_counts[label]).astype(np.uint8)
        if save_imgs:
            save_path = output_path / f"{label}.png"
            cv2.imwrite(str(save_path), templates[label])

    print(f"Templates Extracted")

    return templates


def main():
    extract_templates()
    # match_captcha()

if __name__ == "__main__":
    main()
