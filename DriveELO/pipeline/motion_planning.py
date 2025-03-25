import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from DriveELO.utils.elo_system import update_elo
from DriveELO.utils.glicko_system import update_glicko
from DriveELO.utils.visualization import plot_ratings, plot_samples

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def convert_classification_to_match_results(prediction_result_folder: Path, match_result_folder: Path) -> None:
    # Loop through all .pkl files in the folder
    for csv_file in prediction_result_folder.glob("*.csv"):
        print(f"Processing: {csv_file.name}")

        # Load the DataFrame
        df = pd.read_csv(csv_file) # a dict with scenario_id as keys

        # Get model names from column headers (excluding "Image Index" and "True Label")
        model_name = csv_file.stem

        # Convert to long list of match results
        match_results = []
        for _, row in df.iterrows():
            scenario_id = row['token']
            score = row['score']
            match_results.append([scenario_id, model_name, 1 - score, score])

        # Save to new pkl file
        df_long = pd.DataFrame(match_results, columns=["Test Case", "Model", "Test Case Score", "Model Score"])
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
            model = row["Model"]

            # Initialize ratings if not already set
            if test_case not in test_case_ratings:
                test_case_ratings[test_case] = {'mu': init_mu, 'rd': init_rd}
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
            test_case_data = [(name, values['mu'], values['rd']) for name, values in test_case_ratings.items()]
            model_data = [(name, values['mu'], values['rd']) for name, values in model_ratings.items()]
            test_case_df = pd.DataFrame(test_case_data, columns=["Name", "Rating", "Deviation"])
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
            plot_ratings(test_case_ratings_path, model_ratings_path, rating_figure_path, show_plot=False)
            # sample_figure_path = rating_result_folder.joinpath(f"{method}_sample_{i + 1}.png")
            # plot_samples(dataset_folder, test_case_ratings_path, sample_figure_path, num_samples_per_bin=3)

    return


def main():
    parser = argparse.ArgumentParser(description='Running Classification & Compute Ratings')
    parser.add_argument('--dataset_split', default="val", type=str, help='"train", "val", "mini" ')
    parser.add_argument('--dataset_path',default="/media/shuo/Cappuccino/NAVSIM/", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="./data/motion_planning/NAVSIM/",type=str, help='path to save processed data')
    args = parser.parse_args()

    # Input and Output Data folders
    dataset_folder = Path(args.dataset_path)
    dataset_split_folder = dataset_folder.joinpath(f"{args.dataset_split}/")
    output_folder = Path(args.save_path).joinpath(f"{args.dataset_split}/")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Step 1: Run inference
    prediction_result_folder = output_folder.joinpath("predictions/")
    prediction_result_folder.mkdir(parents=True, exist_ok=True)
    # run_classification(dataset_split_folder, prediction_result_folder, imagenet_class_index, batch_size=512)

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
