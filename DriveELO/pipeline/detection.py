from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import logging

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models


import torchvision
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import os


from DriveELO.utils.elo_system import update_elo
from DriveELO.utils.glicko_system import update_glicko
from DriveELO.utils.visualization import plot_ratings, plot_samples


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

COCO_PATH = "./coco2017"  # Change this if COCO is stored elsewhere

def run_detection(dataset_folder: Path, prediction_result_folder: Path, imagenet_class_index: dict=None, batch_size: int=128) -> None:

    dataset = CocoDetection(
        root=os.path.join(COCO_PATH, "val2017"),
        annFile=os.path.join(COCO_PATH, "annotations/instances_val2017.json"),
        transform=lambda img, target: (F.to_tensor(img), target)
    )

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=lambda batch: tuple(zip(*batch)))

    # Define models
    models_dict = {
        "FasterRCNN-ResNet50": models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
        "RetinaNet-ResNet50": models.detection.retinanet_resnet50_fpn(weights=models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1),
        # "SSD-VGG16": models.detection.ssd300_vgg16(pretrained=True),
        # "FCOS-ResNet50": models.detection.fcos_resnet50_fpn(pretrained=True),
        # "MaskRCNN-ResNet50": models.detection.maskrcnn_resnet50_fpn(pretrained=True),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high') # TF32 is supported on Ampere and newer GPUs (RTX 30/40 series, A100, H100, etc.)

    # Evaluation loop
    with torch.no_grad():
        # Iterate over each model
        for model_name, model in models_dict.items():
            print(f"Processing model: {model_name}")
            model = model.to(device)
            model = torch.compile(model)
            model.eval()

            results = []
            # results_dict = {}
            # Iterate over all images
            for img_idx, (images, labels_gt) in enumerate(tqdm(data_loader, desc=f"Processing images with {model_name}")):
                images = images.to(device)
                labels_gt = labels_gt.to(device)

                # Get filenames for the current batch
                start_idx = img_idx * batch_size
                end_idx = start_idx + len(images)
                batch_filenames = [Path(dataset.imgs[i][0]).name for i in range(start_idx, end_idx)]

                # Batch inference
                try:
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)

                    # Iterate over each image in the batch
                    for i, (filename, output) in enumerate(zip(batch_filenames, outputs)):
                        image_id = targets[i][0]["image_id"] if len(targets[i]) > 0 else -1
                        if image_id == -1:
                            continue

                        # Iterate over each bbox in the image
                        for j in range(len(output["boxes"])):
                            bbox = output["boxes"][j].tolist()
                            score = output["scores"][j].item()
                            label = int(output["labels"][j].item())

                            results.append({
                                "image_id": int(image_id),
                                "category_id": label,
                                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # Convert to COCO format
                                "score": float(score)
                            })

                        true_label = dataset_labels[labels_gt[i].item()]
                        predicted_label = imagenet_labels[predictions[i].item()]

                        results_dict[filename] = {"Image Filename": filename, "True Label": true_label, model_name: predicted_label}

                except Exception as e:
                    logger.error(f"Error processing batch {start_idx} to {end_idx} for model {model_name}: {str(e)}")
                    continue

            del model
            torch.cuda.empty_cache()

            # Convert results to CSV
            df = pd.DataFrame(results_dict.values())
            prediction_file_name = prediction_result_folder.joinpath(f"prediction_{model_name}.pkl")
            df.to_pickle(prediction_file_name)
            print(f"Results saved to {prediction_file_name}")

    return


def convert_classification_to_match_results(prediction_result_folder: Path, match_result_folder: Path) -> None:
    # Loop through all .pkl files in the folder
    for pkl_file in prediction_result_folder.glob("*.pkl"):
        print(f"Processing: {pkl_file.name}")

        # Load the DataFrame
        df = pd.read_pickle(pkl_file)

        # Get model names from column headers (excluding "Image Index" and "True Label")
        model_name = df.columns[2]

        # Convert to long list of match results
        match_results = []
        for _, row in df.iterrows():
            image_name = row["Image Filename"]
            image_label = row["True Label"]

            score = int(row[model_name] == image_label)
            match_results.append([image_label, image_name, model_name, 1 - score, score])

        # Save to new CSV file
        df_long = pd.DataFrame(match_results, columns=["True Label", "Test Case", "Model", "Test Case Score", "Model Score"])
        match_result_path = match_result_folder.joinpath(f"match_{model_name}.pkl")
        df_long.to_pickle(match_result_path)
        print(f"Long-format results saved to {match_result_path}")

    return


def compute_ratings(dataset_folder: Path, match_result_folder: Path, rating_result_folder: Path, num_rounds_list: list,
                    method: str='Elo', init_mu: int=1500, init_rd: int=350, k_max: int=40, k_min: int=10) -> None:
    # Load all .pkl files and concatenate all DataFrames
    pkl_files = sorted(match_result_folder.glob("*.pkl"))
    dfs = [pd.read_pickle(file) for file in pkl_files]
    df = pd.concat(dfs, ignore_index=True)

    # Initialize Ratings
    test_case_ratings = {}
    model_ratings = {}

    # Process each match (test case vs model)
    total_num_rounds = max(num_rounds_list)
    for i in tqdm(range(total_num_rounds), desc=f"Running Match Rounds"):

        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        for _, row in tqdm(df_shuffled.iterrows(), desc=f"Running Matches"):
            test_case = row["Test Case"]
            label = row["True Label"]
            model = row["Model"]

            # Initialize ratings if not already set
            if test_case not in test_case_ratings:
                test_case_ratings[test_case] = {'label': label, 'mu': init_mu, 'rd': init_rd}
            if model not in model_ratings:
                model_ratings[model] = {'mu': init_mu, 'rd': init_rd}

            # Extract current ratings
            rating_test_case = test_case_ratings[test_case]
            rating_model = model_ratings[model]

            # Get match scores
            score_test_case = row["Test Case Score"]
            score_model = row["Model Score"]

            # Update Ratings
            if method == 'Elo':
                k = k_max - i / max(1, total_num_rounds - 1) * (k_max - k_min) # Linear scheduler
                new_rating_test, new_rating_model = update_elo(rating_test_case, rating_model, score_test_case, score_model, k=k)
            elif method == 'Glicko':
                new_rating_test, new_rating_model = update_glicko(rating_test_case, rating_model, score_test_case, score_model)

            # Save updated ratings
            test_case_ratings[test_case] = new_rating_test
            model_ratings[model] = new_rating_model

        if i + 1 in num_rounds_list:
            # Convert Ratings to a DataFrame
            test_case_data = [(values['label'], name, values['mu'], values['rd']) for name, values in test_case_ratings.items()]
            model_data = [(name, values['mu'], values['rd']) for name, values in model_ratings.items()]
            test_case_df = pd.DataFrame(test_case_data, columns=["Label", "Name", "Rating", "Deviation"])
            model_df = pd.DataFrame(model_data, columns=["Name", "Rating", "Deviation"])
            test_case_df = test_case_df.sort_values(by="Rating", ascending=True)
            model_df = model_df.sort_values(by="Rating", ascending=False)

            # Save to an CSV file
            test_case_ratings_path = rating_result_folder.joinpath(f"rating_{method}_test_case_{i + 1}.pkl")
            model_ratings_path = rating_result_folder.joinpath(f"rating_{method}_model_{i + 1}.pkl")
            test_case_df.to_pickle(test_case_ratings_path)
            model_df.to_pickle(model_ratings_path)
            print(f"Ratings saved to {test_case_ratings_path}, {model_ratings_path}")

            # Visualize the Ratings
            rating_figure_path = rating_result_folder.joinpath(f"{method}_dist_{i + 1}.png")
            sample_figure_path = rating_result_folder.joinpath(f"{method}_sample_{i + 1}.png")
            plot_ratings(test_case_ratings_path, model_ratings_path, rating_figure_path, show_plot=False)
            plot_samples(dataset_folder, test_case_ratings_path, sample_figure_path, num_samples_per_bin=3)

    return


def main():
    # Dataset Selection (Change Here)
    dataset_name = "ImageNet"
    dataset_split_name = "train" # "train", "val", "mini"

    # Input and Output Data folders
    dataset_folder = Path(f"/media/shuo/Cappuccino/{dataset_name}/")
    dataset_split_folder = dataset_folder.joinpath(f"{dataset_split_name}/")
    output_folder = Path(f"./data/classification/{dataset_name}/{dataset_split_name}/")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load ImageNet class mappings from a JSON file: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
    with open(dataset_folder.joinpath("imagenet_class_index.json"), "r") as f:
        imagenet_class_index = json.load(f)

    # Step 1: Run inference on ImageNet
    prediction_result_folder = output_folder.joinpath("predictions/")
    prediction_result_folder.mkdir(parents=True, exist_ok=True)
    run_classification(dataset_split_folder, prediction_result_folder, imagenet_class_index, batch_size=512)

    # Step 2: Convert prediction results to match results
    match_result_folder = output_folder.joinpath("matches/")
    match_result_folder.mkdir(parents=True, exist_ok=True)
    convert_classification_to_match_results(prediction_result_folder, match_result_folder)

    # Step 3: Compute Ratings based on match results
    method = 'Glicko' # 'Elo'
    num_rounds_list = [1, 2, 3, 4, 5, 10]
    rating_result_folder = output_folder.joinpath("ratings/")
    rating_result_folder.mkdir(parents=True, exist_ok=True)
    compute_ratings(dataset_split_folder, match_result_folder, rating_result_folder, num_rounds_list, method=method)

if __name__ == "__main__":
    main()
