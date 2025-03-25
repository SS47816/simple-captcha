import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.captcha import Captcha


def main():
    parser = argparse.ArgumentParser(description='identify captcha')
    parser.add_argument('--im_path', default="./data/input/input100.jpg", type=str, help='path to the input .jpg image file')
    parser.add_argument('--save_path',default="./data/prediction/output100.txt", type=str, help='path to save the result .txt files')
    args = parser.parse_args()

    captcha = Captcha()
    captcha(args.im_path, args.save_path)
    print(f"Prediction result for {args.im_path} has been saved to {args.save_path}")

if __name__ == "__main__":
    main()
