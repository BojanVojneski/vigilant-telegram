"""Make a small, descriptive report from the already-scored comments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emotion_labels import EMOTION_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = PROJECT_ROOT / "input" / "final_scored_comments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results"
GROUP_ORDER = ["negative", "positive", "ambiguous", "neutral"]
BLUE = "#4C78A8"
ORANGE = "#F58518"
GROUP_LABELS = {
    "negative": "Negativna",
    "positive": "Pozitivna",
    "ambiguous": "Nejasna",
    "neutral": "Neutralna",
}
def run_simple_analysis(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    emotions_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write simple counts, averages, and charts; no hypothesis tests."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Ne e pronajdena datotekata so oceneti komentari: {input_path}. "
            "Stavete ja CSV-datotekata na ovaa lokacija."
        )

    emotion_file = (
        Path(emotions_path)
        if emotions_path is not None
        else PROJECT_ROOT / "input" / "emotions.txt"
    )
    emotions = [
        line.strip()
        for line in emotion_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not emotions or len(emotions) != len(set(emotions)):
        raise ValueError(
            f"Listata na emocii mora da ima unikatni stavki: {emotion_file}"
        )
    human_columns = [f"human_{emotion}" for emotion in emotions]
    required = {
        *emotions,
        *human_columns,
        "glavna_emocija",
        "grupa",
        "dol",
        "vulg_any",
        "pol",
        "sub",
        "subreddit",
    }
    data = pd.read_csv(input_path)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Vo datotekata nedostasuvaat koloni: {sorted(missing)}"
        )
    if data.empty:
        raise ValueError("Datotekata so oceneti komentari e prazna")

    human_scores = data[human_columns]
    has_human_labels = human_scores.sum(axis=1) > 0
    human_dominant = human_scores.loc[has_human_labels].idxmax(axis=1)
    human_dominant = human_dominant.str.removeprefix("human_")
    model_counts = data["glavna_emocija"].value_counts().reindex(emotions, fill_value=0)
    human_counts = human_dominant.value_counts().reindex(emotions, fill_value=0)
    emotion_table = pd.DataFrame(
        {
            "Emocija": [EMOTION_LABELS.get(emotion, emotion) for emotion in emotions],
            "Model_komentari": model_counts.to_numpy(),
            "Model_procent": model_counts.to_numpy() / len(data) * 100,
            "Izvorni_rateri_komentari": human_counts.to_numpy(),
            "Izvorni_rateri_procent": (
                human_counts.to_numpy() / len(human_dominant) * 100
            ),
        }
    )

    group_table = (
        data.groupby("grupa", observed=True)
        .agg(
            Broj_komentari=("grupa", "size"),
            Prosechno_zborovi=("dol", "mean"),
            Mediana_zborovi=("dol", "median"),
            Komentari_so_zbor_od_listata_za_psovki=("vulg_any", "mean"),
            Prosechen_ton=("pol", "mean"),
            Prosechna_subjektivnost=("sub", "mean"),
        )
        .reindex(GROUP_ORDER)
        .dropna(subset=["Broj_komentari"])
        .reset_index()
    )
    group_table["Emocionalna_grupa"] = group_table["grupa"].map(GROUP_LABELS)
    group_table = group_table.drop(columns="grupa").rename(
        columns={
            "Komentari_so_zbor_od_listata_za_psovki": (
                "Komentari_so_zbor_od_listata_za_psovki_procent"
            )
        }
    )
    group_table["Komentari_so_zbor_od_listata_za_psovki_procent"] *= 100

    subreddit_table = (
        data.groupby("subreddit", observed=True)
        .agg(
            Broj_komentari=("subreddit", "size"),
            Prosechen_neneutralen_rezultat=("neutral", lambda scores: 1 - scores.mean()),
        )
        .query("Broj_komentari >= 200")
        .sort_values("Broj_komentari", ascending=False)
        .head(10)
        .sort_values("Prosechen_neneutralen_rezultat")
        .reset_index()
    )
    subreddit_table["Prosechen_neneutralen_rezultat"] *= 100
    subreddit_table = subreddit_table.rename(
        columns={
            "subreddit": "Reddit_zaednica",
            "Prosechen_neneutralen_rezultat": "Prosechen_neneutralen_rezultat_procent",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    emotion_path = output_dir / "chestota_emocii.csv"
    group_path = output_dir / "rezime_emocionalni_grupi.csv"
    subreddit_path = output_dir / "rezime_reddit_zaednici.csv"
    emotion_table.to_csv(emotion_path, index=False)
    group_table.to_csv(group_path, index=False)
    subreddit_table.to_csv(subreddit_path, index=False)

    top_emotions = (
        emotion_table.assign(
            vkupno=lambda table: (
                table["Model_komentari"] + table["Izvorni_rateri_komentari"]
            ),
        )
        .set_index("Emocija")
        .nlargest(10, "vkupno")
        .sort_values("vkupno")
    )
    figure_paths = {}

    fig, axis = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top_emotions))
    height = 0.38
    axis.barh(
        y - height / 2,
        top_emotions["Model_procent"],
        height,
        label="Model",
        color=BLUE,
    )
    axis.barh(
        y + height / 2,
        top_emotions["Izvorni_rateri_procent"],
        height,
        label="Izvorni ocenuvachi",
        color=ORANGE,
    )
    axis.set_yticks(y, top_emotions.index)
    axis.set_xlabel("Komentari (%)")
    axis.set_title("Najchesti dominantni emocii")
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    figure_paths["emotion_frequency"] = output_dir / "chestota_emocii.png"
    fig.savefig(figure_paths["emotion_frequency"], dpi=140, bbox_inches="tight")
    plt.close(fig)

    charts = [
        (
            "Komentari_so_zbor_od_listata_za_psovki_procent",
            "Komentari so zbor od listata za psovki (%)",
            "Komentari (%)",
            None,
            "psovki",
        ),
        ("Mediana_zborovi", "Tipichna dolzhina na komentar", "Mediana na zborovi", None, "dolzhina"),
        ("Prosechen_ton", "Prosechen ton na tekstot", "Prosechen ton (-1 negativno, 1 pozitivno)", None, "ton"),
        (
            "Prosechna_subjektivnost",
            "Prosechna subjektivnost na tekstot",
            "Rezultat (0 do 1)",
            1,
            "subjektivnost",
        ),
    ]
    for column, title, ylabel, limit, name in charts:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        bars = axis.bar(
            group_table["Emocionalna_grupa"],
            group_table[column],
            color=BLUE,
        )
        axis.set_title(title)
        axis.set_xlabel("Emocionalna grupa od modelot")
        axis.set_ylabel(ylabel)
        if limit is not None:
            axis.set_ylim(0, limit)
        if column == "Prosechen_ton":
            axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, group_table[column]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}" if abs(value) < 1 else f"{value:.1f}",
                ha="center",
                va="bottom",
            )
        fig.tight_layout()
        figure_paths[name] = output_dir / f"{name}.png"
        fig.savefig(figure_paths[name], dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(
        subreddit_table["Reddit_zaednica"],
        subreddit_table["Prosechen_neneutralen_rezultat_procent"],
        color=BLUE,
    )
    axis.set_xlabel("Prosechen neneutralen rezultat na modelot (%)")
    axis.set_title("Najgolemi Reddit zaednici")
    axis.set_xlim(0, 100)
    axis.grid(axis="x", alpha=0.2)
    for bar, value in zip(
        bars,
        subreddit_table["Prosechen_neneutralen_rezultat_procent"],
    ):
        axis.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%",
            va="center",
        )
    fig.tight_layout()
    figure_paths["subreddits"] = output_dir / "reddit_zaednici.png"
    fig.savefig(figure_paths["subreddits"], dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Sumirani komentari: {len(data):,}")
    print(f"Komentari so oznaki od izvornite ocenuvachi: {len(human_dominant):,}")
    print(f"Tabeli i ednostavni grafikoni: {output_dir}")
    return {
        "emotion_table": emotion_path,
        "group_table": group_path,
        "subreddit_table": subreddit_path,
        **figure_paths,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kreiraj opisni tabeli i grafikoni za emociite vo komentarite."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--emotions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    run_simple_analysis(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        emotions_path=arguments.emotions,
    )
