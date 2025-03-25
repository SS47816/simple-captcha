import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.captcha import Captcha


class TestCaptcha(unittest.TestCase):
    def setUp(self):
        self.captcha = Captcha()

    def test_performance(self):
        input_path = Path("./data/input/")
        label_path = Path("./data/output/")
        output_path = Path("./data/prediction/")
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all .jpg files in the folder
        results = {}
        image_files = list(input_path.glob("*.jpg"))
        for im_path in image_files:
            new_stem = im_path.stem.replace("input", "output")
            save_path = output_path / (new_stem + ".txt")
            self.captcha(im_path, save_path)

            # Validate wrt the ground truth label file
            label_file = label_path / (new_stem + ".txt")
            if label_file.exists():
                with open(label_file, "r") as f:
                    labels = f.read().strip()
                with open(save_path, "r") as f:
                    preds = f.read().strip()
                results[im_path.stem] = 1 if preds == labels else 0
            else:
                # print(f"Warning: Label file {label_file} not provided, skipped this data.")
                continue

        correct = sum(1 for value in results.values() if value == 1)
        accuracy = correct / len(results) if len(results) > 0 else 0

        print(f"The predicton accuracy is {accuracy * 100:.2f}%")


if __name__ == '__main__':
    unittest.main()
