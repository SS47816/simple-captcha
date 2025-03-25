import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from DriveELO.utils.elo_system import update_elo
from DriveELO.utils.glicko_system import update_glicko
from DriveELO.utils.visualization import plot_ratings, plot_samples

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def run_classification(dataset_folder: Path, prediction_result_folder: Path, imagenet_class_index: dict=None, batch_size: int=128) -> None:
    # Define ImageNet normalization and resizing
    transform = transforms.Compose(
        [
            transforms.Resize(256),  # Resize shorter side to 256
            transforms.CenterCrop(224),  # Crop to 224x224 for pretrained models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(dataset_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Load ImageNet class labels {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
    dataset_labels = {i: class_name for class_name, i in dataset.class_to_idx.items()} # "0": "n01484850" (by dataset seq)
    imagenet_labels = {int(k): v[0] for k, v in imagenet_class_index.items()} # "0": "n01440764" (by imagenet definition)
    imagenet_classes = {v[0]: v[1] for k, v in imagenet_class_index.items()} # "n01440764": "tench" (by imagenet definition)

    # Define models
    models_dict = {
        "ConvNeXt-Large": models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1),
        # "ConvNeXt-Base": models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
        "ConvNeXt-Tiny": models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        # "MaxViT-T": models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1),
        "Swin-B": models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1),
        # "Swin-S": models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1),
        "Swin-T": models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1),
        "ViT-L-32": models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1),
        "ViT-B-16": models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),

        # "EfficientNet-L2": models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1),
        # "EfficientNet-B7": models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1),
        "EfficientNet-B0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "ResNext-101": models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1),
        "ResNext-50": models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
        "ResNet-152": models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1),
        # "ResNet-101": models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1),
        # "ResNet-50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        # "ResNet-34": models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1),
        "ResNet-18": models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),

        "RegNet-Y-800MF": models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1),
        # "RegNet-Y-16GF": models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1),
        "RegNet-X-8GF": models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V1),
        "DenseNet-121": models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
        "Inception-v3": models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
        "MobileNet-v2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        "VGG-16": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        "ShuffleNetV2-x1-0": models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
        "SqueezeNet1-0": models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1),
        "AlexNet": models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
    }

    # models_dict = {
    #     "ConvNeXt-Large": models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1),
    #     "Swin-B": models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1),
    #     "ViT-L-32": models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1),
    #     "EfficientNet-B0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    #     "ResNet-101": models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1),
    #     "DenseNet-121": models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
    #     "Inception-v3": models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
    #     "MobileNet-v2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
    #     "VGG-16": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
    #     "AlexNet": models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
    # }

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

            results_dict = {}
            # Iterate over all images
            for batch_id, (images, labels_gt) in enumerate(tqdm(data_loader, desc=f"Processing images with {model_name}")):
                images = images.to(device)
                labels_gt = labels_gt.to(device)

                # Get filenames for the current batch
                start_idx = batch_id * batch_size
                end_idx = start_idx + len(images)
                batch_filenames = [Path(dataset.imgs[i][0]).name for i in range(start_idx, end_idx)]

                # Batch inference
                try:
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)

                    # Iterate over each image in the batch
                    for i, filename in enumerate(batch_filenames):
                        true_label = dataset_labels[labels_gt[i].item()]
                        predicted_label = imagenet_labels[predictions[i].item()]

                        results_dict[filename] = {"Image Filename": filename, "True Label": true_label, model_name: predicted_label}

                except Exception as e:
                    logger.error(f"Error processing batch {start_idx} to {end_idx} for model {model_name}: {str(e)}")
                    continue

            del model
            torch.cuda.empty_cache()

            # Convert results to pkl
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

        # Save to new pkl file
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

            # Save to an pkl file
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
    parser = argparse.ArgumentParser(description='Running Classification & Compute Ratings')
    parser.add_argument('--dataset_split', default="train", type=str, help='"train", "val", "mini" ')
    parser.add_argument('--dataset_path',default="/media/shuo/Cappuccino/ImageNet/", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="./data/classification/ImageNet/",type=str, help='path to save processed data')
    args = parser.parse_args()

    # Input and Output Data folders
    dataset_folder = Path(args.dataset_path)
    dataset_split_folder = dataset_folder.joinpath(f"{args.dataset_split}/")
    output_folder = Path(args.save_path).joinpath(f"{args.dataset_split}/")
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
