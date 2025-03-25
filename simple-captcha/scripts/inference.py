from pathlib import Path

import cv2
import numpy as np


def load_dataset(data_folder="./data"):
    input_path = Path(data_folder).joinpath("input")
    label_path = Path(data_folder).joinpath("output")

    dataset = {}

    # Get all .jpg files in the folder
    image_files = list(input_path.glob("*.jpg"))

    for image_path in image_files:
        key = image_path.stem
        new_stem = image_path.stem.replace("input", "output")
        label_file = label_path / (new_stem + ".txt")

        # Read label file
        if label_file.exists():
            with open(label_file, "r") as f:
                labels = f.read().strip()
        else:
            print(f"Warning: Label file {label_file} not found.")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        dataset[key] = (image, labels)

    return dataset

def extract_templates(dataset, bboxes, output_folder="./data/template/", save_imgs=False):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dictionary to store accumulated templates and counts
    templates = {}
    template_counts = {}
    for key, (image, labels) in dataset.items():

        _, image_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Process each bounding box and corresponding character label
        for i, (x, y, w, h) in enumerate(bboxes):
            char_crop = image_thresh[y:y+h, x:x+w].astype(np.float32)
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


def identify_captcha(templates, bboxes, input_folder="./data/input/", output_folder="./data/prediction/"):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all .jpg files in the folder
    image_files = list(input_path.glob("*.jpg"))

    for image_path in image_files:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        identified_chars = []
        # Process each bounding box and match with templates
        for i, (x, y, w, h) in enumerate(bboxes):
            char_crop = thresh[y:y+h, x:x+w]
            best_match_label = None
            matching_scores = {}
            # Compare cropped character with each template
            for label, template in templates.items():
                result = cv2.matchTemplate(char_crop, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                matching_scores[label] = max_val

            # Append the identified label for the character
            best_match_label = max(matching_scores, key=matching_scores.get)
            identified_chars.append(best_match_label)

        # Save identified characters in a text file
        output_file = output_path / (image_path.stem + ".txt")
        with open(str(output_file), "w") as f:
            f.write("".join(identified_chars))

    print(f"Prediction results saved in {output_folder}")


def main():
    # Define fixed bounding boxes (x, y, width, height)
    bboxes = [(5 + 9*i, 11, 8, 10) for i in range(5)]
    num_chars = 5

    dataset_train = load_dataset("./data")
    templates = extract_templates(dataset_train, bboxes, output_folder="./data/template/", save_imgs=False)
    identify_captcha(templates, bboxes)

if __name__ == "__main__":
    main()
