from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_ratings(test_case_ratings_path: Path, model_ratings_path: Path, figure_path: Path, show_plot: bool=False) -> None:
    # Load Ratings
    test_cases_df = pd.read_pickle(test_case_ratings_path)  # Test cases
    models_df = pd.read_pickle(model_ratings_path)  # Models
    num_test_cases = test_cases_df.shape[0]
    num_models = models_df.shape[0]

    # Sort test case ratings for cumulative percentage calculation
    test_case_ratings = test_cases_df["Rating"].values
    sorted_ratings = np.sort(test_case_ratings)
    cumulative_percent = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings) * 100

    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Histogram (Test Case Ratings)
    rating_start = (sorted_ratings.min() // 100) * 100
    rating_end = ((sorted_ratings.max() // 100) + 1) * 100 + 1
    rating_bins = np.arange(start=rating_start, stop=rating_end, step=50)
    color_hist = 'skyblue'
    ax1.hist(test_case_ratings, bins=rating_bins, color=color_hist, alpha=0.7, edgecolor='black', label="Test Case Ratings")
    ax1.set_xlim(rating_start-200, rating_end)
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Number of Test Cases", color=color_hist)
    ax1.tick_params(axis='y', labelcolor=color_hist)

    # Secondary Y-axis for Cumulative Percentage
    ax2 = ax1.twinx()
    color_curve = 'gold'
    ax2.plot(sorted_ratings, cumulative_percent, color=color_curve, linestyle='-', linewidth=2, label="Cumulative %")
    ax2.set_ylabel("Cumulative Percentage", color=color_curve)
    ax2.tick_params(axis='y', labelcolor=color_curve)

    # Vertical lines for Model Ratings
    model_colors = cm.get_cmap('tab20', num_models).colors
    for name, rating, color in zip(models_df["Name"].values, models_df["Rating"].values, model_colors):
        ax1.axvline(rating, color=color, linestyle='dashed', linewidth=2, label=name)

    # Title & Legends
    ax1.set_title("Test Case & Model Rating Distributions")
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")

    plt.savefig(figure_path, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_samples(dataset_path: Path, test_case_ratings_path: Path, figure_path: Path, num_samples_per_bin: int=3, show_plot: bool=False):
    df = pd.read_pickle(test_case_ratings_path)

    # Get test case ratings and define bins
    test_case_ratings = df["Rating"].values
    bin_width = 200
    rating_start = (test_case_ratings.min() // bin_width) * bin_width
    rating_end = ((test_case_ratings.max() // bin_width) + 1) * bin_width + 1
    rating_bins = np.arange(start=rating_start, stop=rating_end, step=bin_width)

    # Compute percentiles for each bin
    sorted_ratings = np.sort(test_case_ratings)
    # percentiles = [(np.sum(sorted_ratings <= bin_max) / len(sorted_ratings) * 100) for bin_max in rating_bins[1:]]
    percentiles = [(np.sum(sorted_ratings > bin_min) / len(sorted_ratings) * 100) for bin_min in rating_bins[:-1]]

    # Dictionary to store selected images
    selected_images = {}

    # Group images by rating bins and sample
    for i in range(len(rating_bins) - 1):
        bin_min, bin_max = rating_bins[i], rating_bins[i + 1]
        bin_df = df[(df["Rating"] >= bin_min) & (df["Rating"] < bin_max)]
        sampled = bin_df.sample(min(num_samples_per_bin, len(bin_df)), random_state=42)

        if not sampled.empty:
            selected_images[bin_min] = sampled[["Name", "Label"]].values.tolist()

    # Figure settings
    num_cols = len(selected_images)
    num_rows = max(len(images) for images in selected_images.values()) if selected_images else 1
    fig_width = max(7, num_cols * 1.5)
    fig_height = max(3, num_rows * 1.5)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height))

    # Ensure axes is always a 2D array (prevents indexing issues)
    axes = np.atleast_2d(axes)

    # Plot images
    for col, (bin_min, images) in enumerate(selected_images.items()):
        for row, (img_name, img_label) in enumerate(images):
            img_path = dataset_path.joinpath(img_label, img_name)

            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is None:  # Handle OpenCV read errors
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                axes[row, col].imshow(img)
                axes[row, col].axis("off")

    # Titles for rating bins
    for col, (bin_min, percentile) in enumerate(zip(selected_images.keys(), percentiles)):
        axes[0, col].set_title(f"{int(bin_min)}-{int(bin_min + bin_width)}\nTop {percentile:.1f}% Hard", fontsize=10)

    plt.subplots_adjust(hspace=0.05, wspace=0.1)

    plt.savefig(figure_path)  # Save first
    if show_plot:
        plt.show()
