from pathlib import Path

import cv2
import numpy as np


class Captcha(object):
    def __init__(self):
        self._bboxes = [(5 + 9*i, 11, 8, 10) for i in range(5)]
        self._num_chars = 5
        self._train_dataset = {}
        self._templates = {}
        self._load_train_dataset("./data")
        self._extract_templates()

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        image = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        identified_chars = []
        # Process each bounding box and match with templates
        for i, (x, y, w, h) in enumerate(self._bboxes):
            char_crop = thresh[y:y+h, x:x+w]
            best_match_label = None
            matching_scores = {}
            # Compare cropped character with each template
            for label, template in self._templates.items():
                result = cv2.matchTemplate(char_crop, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                matching_scores[label] = max_val

            # Append the identified label for the character
            best_match_label = max(matching_scores, key=matching_scores.get)
            identified_chars.append(best_match_label)

        # Save identified characters in a text file
        with open(str(save_path), "w") as f:
            f.write("".join(identified_chars))

        return

    def _load_train_dataset(self, data_folder: str):
        """
        Load the training dataset from the specified folder.

        This method reads image files from the 'input' subfolder and their corresponding
        label files from the 'output' subfolder. It populates the _train_dataset
        dictionary with image-label pairs.

        Args:
            data_folder (str): The path to the folder containing 'input' and 'output' subfolders.

        Returns:
            None
        """
        input_path = Path(data_folder).joinpath("input")
        label_path = Path(data_folder).joinpath("output")

        # Get all .jpg files in the folder
        image_files = list(input_path.glob("*.jpg"))

        self._train_dataset = {}
        for image_path in image_files:
            key = image_path.stem
            new_stem = image_path.stem.replace("input", "output")
            label_file = label_path / (new_stem + ".txt")

            # Read label file
            if label_file.exists():
                with open(label_file, "r") as f:
                    labels = f.read().strip()
            else:
                # print(f"Warning: Label file {label_file} not provided, skipped this data.")
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            self._train_dataset[key] = (image, labels)

        return

    def _extract_templates(self, output_folder: str="./data/template/", save_imgs: bool=False):
        """
        Extract and create averaged templates for each character from the training dataset.

        This method processes the training dataset to create averaged templates for each unique
        character. It thresholds each image, crops characters based on predefined bounding boxes,
        and accumulates them to create an average template for each character.

        Args:
            output_folder (str): The path where template images will be saved if save_imgs is True.
                                 Defaults to "./data/template/".
            save_imgs (bool): If True, saves the generated templates as image files.
                              Defaults to False.

        Returns:
            None
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Dictionary to store accumulated templates and counts
        templates = {}
        template_counts = {}
        for key, (image, labels) in self._train_dataset.items():

            _, image_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Process each bounding box and corresponding character label
            for i, (x, y, w, h) in enumerate(self._bboxes):
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
        self._templates = templates

        return
