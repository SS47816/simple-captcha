from pathlib import Path

import cv2
import numpy as np


def extract_templates(input_folder="./data/input/", label_folder="./data/output/", output_folder="./data/template/", num_chars=5):
    input_path = Path(input_folder)
    label_path = Path(label_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define fixed bounding boxes (x, y, width, height)
    bboxes = [(5 + 9*i, 11, 8, 10) for i in range(num_chars)]

    # Get all .jpg files in the folder
    image_files = list(input_path.glob("*.jpg"))

    for image_path in image_files:
        # Find the corresponding label file
        label_file = label_path / (image_path.stem.replace("input", "output") + ".txt")
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
            char_crop = thresh[y:y+h, x:x+w]

            filename = f"{labels[i]}_{image_path.stem}_char{i}.png"
            save_path = output_path / filename

            # Save the template
            cv2.imwrite(str(save_path), char_crop)

    print(f"Extracted templates saved in {output_folder}")


def match_captcha(new_captcha_path, template_folder="templates"):
    # Load new CAPTCHA image
    new_image = cv2.imread(new_captcha_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess
    _, new_thresh = cv2.threshold(new_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (characters in the new CAPTCHA)
    contours, _ = cv2.findContours(new_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    matched_text = ""

    # Loop over extracted characters
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_crop = new_thresh[y:y+h, x:x+w]
        char_crop = cv2.resize(char_crop, (28, 28))

        best_match = None
        highest_score = 0

        # Compare with stored templates
        for template_name in os.listdir(template_folder):
            template_path = os.path.join(template_folder, template_name)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            # Template matching
            res = cv2.matchTemplate(char_crop, template, cv2.TM_CCOEFF_NORMED)
            score = np.max(res)

            if score > highest_score:
                highest_score = score
                best_match = template_name[5]  # Extract char from filename (e.g., 'char_A.png')

        matched_text += best_match

    return matched_text




def main():
    extract_templates()

    # Test the function on a new CAPTCHA
    # captcha_result = match_captcha("new_captcha.png")
    # print("Recognized CAPTCHA:", captcha_result)

if __name__ == "__main__":
    main()
