"""Reproducible analysis of morphological differences among penguin species.

Run from the project repository root:
    python analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "penguins.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

SPECIES_ORDER = ["Adelie", "Chinstrap", "Gentoo"]
COLORS = {
    "Adelie": "#2A9D8F",
    "Chinstrap": "#E76F51",
    "Gentoo": "#E9C46A",
}
FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
FEATURE_LABELS = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
}


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the public dataset and retain complete morphometric records."""
    df = pd.read_csv(path)
    required = {"species", *FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = df.dropna(subset=["species", *FEATURES]).copy()
    clean = clean[clean["species"].isin(SPECIES_ORDER)]
    clean["species"] = pd.Categorical(
        clean["species"], categories=SPECIES_ORDER, ordered=True
    )
    return clean.sort_values(["species", "bill_length_mm"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and standard deviation for each feature by species."""
    summary = (
        df.groupby("species", observed=True)[FEATURES]
        .agg(["mean", "std", "count"])
        .round(2)
    )
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    return summary


def nearest_centroid_loo(df: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate accuracy with leave-one-out nearest-centroid classification.

    Features are standardized within each training fold to avoid using the
    held-out observation when computing the scale.
    """
    x = df[FEATURES].to_numpy(dtype=float)
    y = df["species"].astype(str).to_numpy()
    classes = np.array(SPECIES_ORDER)
    predictions = np.empty(len(df), dtype=object)

    for held_out in range(len(df)):
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[held_out] = False
        x_train, y_train = x[train_mask], y[train_mask]
        x_test = x[held_out]

        center = x_train.mean(axis=0)
        scale = x_train.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        z_train = (x_train - center) / scale
        z_test = (x_test - center) / scale

        centroids = np.vstack(
            [z_train[y_train == species].mean(axis=0) for species in classes]
        )
        distances = np.linalg.norm(centroids - z_test, axis=1)
        predictions[held_out] = classes[int(np.argmin(distances))]

    accuracy = float(np.mean(predictions == y))
    return accuracy, y, predictions


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Build a confusion matrix with species in a stable order."""
    table = pd.crosstab(
        pd.Categorical(y_true, categories=SPECIES_ORDER),
        pd.Categorical(y_pred, categories=SPECIES_ORDER),
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )
    return table.astype(int)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)


def plot_bill_scatter(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for species in SPECIES_ORDER:
        subset = df[df["species"] == species]
        ax.scatter(
            subset["bill_length_mm"],
            subset["bill_depth_mm"],
            s=44,
            alpha=0.72,
            color=COLORS[species],
            edgecolor="white",
            linewidth=0.55,
            label=species,
        )
        centroid = subset[["bill_length_mm", "bill_depth_mm"]].mean()
        ax.scatter(
            centroid["bill_length_mm"],
            centroid["bill_depth_mm"],
            marker="X",
            s=180,
            color=COLORS[species],
            edgecolor="#222222",
            linewidth=1.1,
            zorder=5,
        )

    ax.set_title("Bill dimensions separate the three penguin species", pad=16, fontsize=15)
    ax.set_xlabel("Bill length (mm)")
    ax.set_ylabel("Bill depth (mm)")
    style_axis(ax)
    ax.legend(title="Species", frameon=False, loc="best")
    ax.text(
        0.01,
        0.01,
        "Large X marks show species centroids.",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout()
    output = IMAGE_DIR / "bill_scatter.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_trait_profiles(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()

    for ax, feature in zip(axes, FEATURES):
        grouped = [
            df.loc[df["species"] == species, feature].dropna()
            for species in SPECIES_ORDER
        ]
        box = ax.boxplot(
            grouped,
            patch_artist=True,
            widths=0.58,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.45},
        )
        for patch, species in zip(box["boxes"], SPECIES_ORDER):
            patch.set_facecolor(COLORS[species])
            patch.set_edgecolor("#333333")
            patch.set_alpha(0.85)

        ax.set_title(FEATURE_LABELS[feature], fontsize=12, loc="left")
        ax.set_xticks(range(1, 4), SPECIES_ORDER, rotation=12)
        style_axis(ax)

    fig.suptitle(
        "Morphological profiles differ by species",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Boxes show the interquartile range; lines show medians; dots show outliers.",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    output = IMAGE_DIR / "trait_profiles.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_confusion(matrix: pd.DataFrame, accuracy: float) -> Path:
    values = matrix.to_numpy()
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=values.max())

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value > values.max() * 0.55 else "#1D3557"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=12)

    ax.set_xticks(range(len(SPECIES_ORDER)), SPECIES_ORDER, rotation=15)
    ax.set_yticks(range(len(SPECIES_ORDER)), SPECIES_ORDER)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("Actual species")
    ax.set_title(f"Leave-one-out confusion matrix ({accuracy:.1%} accuracy)", pad=14)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    output = IMAGE_DIR / "confusion_matrix.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    df = load_and_clean_data(DATA_PATH)
    summary = summarize(df)
    accuracy, y_true, y_pred = nearest_centroid_loo(df)
    matrix = confusion_table(y_true, y_pred)

    summary.to_csv(OUTPUT_DIR / "species_summary.csv")
    matrix.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    metrics = {
        "dataset": "Palmer Penguins",
        "license": "CC0 1.0",
        "records_after_complete_case_filter": int(len(df)),
        "species_count": int(df["species"].nunique()),
        "features": FEATURES,
        "validation": "leave-one-out nearest-centroid classifier",
        "accuracy": round(accuracy, 4),
        "accuracy_percent": f"{accuracy:.1%}",
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    plot_bill_scatter(df)
    plot_trait_profiles(df)
    plot_confusion(matrix, accuracy)

    print(f"Clean sample size: {len(df)}")
    print(f"Species counts:\n{df['species'].value_counts().sort_index().to_string()}")
    print(f"Leave-one-out accuracy: {accuracy:.1%}")
    print(f"Confusion matrix:\n{matrix.to_string()}")
    print(f"Outputs written to: {OUTPUT_DIR} and {IMAGE_DIR}")


if __name__ == "__main__":
    main()
