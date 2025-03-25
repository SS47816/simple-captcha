from pathlib import Path

import pandas as pd


def main():
    # Dataset Selection
    dataset_name = "Waymo" # "ImageNet", "MMLU", "Waymo"
    dataset_split_name = "val" # "train", "val", "test", "mini"
    result_folder = "predictions" # "predictions", "matches", "ratings"

    # Set the folder containing the .pkl files
    if dataset_name == "ImageNet":
        pkl_folder = Path(f"./data/classification/{dataset_name}/{dataset_split_name}/{result_folder}/")
    elif dataset_name == "MMLU":
        pkl_folder = Path(f"./data/question_answering/{dataset_name}/{dataset_split_name}/{result_folder}/")
    elif dataset_name == "Waymo":
        pkl_folder = Path(f"./data/motion_prediction/{dataset_name}/{dataset_split_name}/{result_folder}/")

    # Find all .pkl files in the folder
    pkl_files = sorted(pkl_folder.glob("*.pkl"))

    # Loop through each .pkl file and display its first few rows
    for pkl_file in pkl_files:
        print(f"Inspecting: {pkl_file.name}")
        df = pd.read_pickle(pkl_file)

        if isinstance(df, pd.DataFrame):
            print(df.head(10))  # Show first 10 rows
        elif isinstance(df, dict):
            # print(df.keys())
            print(list(df.items())[0])
        else:
            print(type(df))
        print("=" * 80)  # Separator

if __name__ == "__main__":
    main()
