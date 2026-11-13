"""Make a descriptive report grouped by VerA ML predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emotion_labels import EMOTION_LABELS
from plot_style import (
    BLUE,
    GOLD,
    GRID,
    INK,
    ORANGE,
    RED,
    SLATE,
    TEAL,
    apply_plot_style,
)
from simple_ml import predict_scores

apply_plot_style()

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = PROJECT_ROOT / "input" / "final_scored_comments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results"
DEFAULT_MODEL = PROJECT_ROOT / "results" / "ml" / "model.joblib"
SENTIMENT_MAPPING = (
    PROJECT_ROOT / "data" / "simplified" / "sentiment_mapping.json"
)
GROUP_ORDER = ["negative", "positive", "ambiguous", "neutral"]
GROUP_LABELS = {
    "negative": "Negativna",
    "positive": "Pozitivna",
    "ambiguous": "Nejasna",
    "neutral": "Neutralna",
}
GROUP_COLORS = {
    "negative": RED,
    "positive": TEAL,
    "ambiguous": GOLD,
    "neutral": SLATE,
}


def _style_axis(axis, grid_axis: str = "x") -> None:
    axis.set_axisbelow(True)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB4BF")
    axis.tick_params(colors=INK, labelsize=9)


def _save_figure(
    fig,
    path: Path,
    rect: tuple[float, float, float, float] = (0, 0, 1, 1),
) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_emotion_groups(emotions: list[str]) -> dict[str, str]:
    mapping = json.loads(SENTIMENT_MAPPING.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError(
            f"Nevalidno mapiranje na emocionalni grupi: {SENTIMENT_MAPPING}"
        )

    groups = {"neutral": "neutral"}
    for group, members in mapping.items():
        if group not in {"negative", "positive", "ambiguous"} or not isinstance(
            members, list
        ):
            raise ValueError(f"Nevalidna emocionalna grupa vo: {SENTIMENT_MAPPING}")
        for emotion in members:
            if (
                not isinstance(emotion, str)
                or emotion not in emotions
                or emotion in groups
            ):
                raise ValueError(
                    f"Nevalidna ili povtorena emocija vo: {SENTIMENT_MAPPING}"
                )
            groups[emotion] = group

    if set(groups) != set(emotions):
        raise ValueError(
            f"Mapiranjeto na grupi ne gi opfakja site emocii: {SENTIMENT_MAPPING}"
        )
    return groups


def run_simple_analysis(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    emotions_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL,
) -> dict[str, Path]:
    """Write simple summaries grouped by VerA's dominant emotion predictions."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Ne e pronajdena datotekata so anotirani komentari: {input_path}. "
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
    required = {
        "text",
        *(f"human_{emotion}" for emotion in emotions),
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
    if (
        data["text"].isna().any()
        or data["text"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Site komentari mora da sodrzhat tekst")

    scores, model_emotions, _ = predict_scores(
        data["text"].astype(str).tolist(),
        model_path,
    )
    if model_emotions != emotions:
        raise ValueError(
            "Listata na emocii vo modelot ne se sovpagja so input/emotions.txt"
        )
    emotion_groups = _load_emotion_groups(emotions)
    data[emotions] = pd.DataFrame(scores, index=data.index, columns=emotions)
    data["glavna_emocija"] = [emotions[index] for index in scores.argmax(axis=1)]
    data["grupa"] = data["glavna_emocija"].map(emotion_groups)
    if data["grupa"].isna().any():
        raise ValueError("Modelot proizvede emocija bez emocionalna grupa")

    human_columns = [f"human_{emotion}" for emotion in emotions]
    human_scores = data[human_columns]
    has_human_labels = human_scores.sum(axis=1) > 0
    human_dominant = human_scores.loc[has_human_labels].idxmax(axis=1)
    human_dominant = human_dominant.str.removeprefix("human_")
    model_counts = (
        data["glavna_emocija"].value_counts().reindex(emotions, fill_value=0)
    )
    human_counts = human_dominant.value_counts().reindex(emotions, fill_value=0)
    emotion_table = pd.DataFrame(
        {
            "Emocija": [EMOTION_LABELS.get(emotion, emotion) for emotion in emotions],
            "VerA_ML_komentari": model_counts.to_numpy(),
            "VerA_ML_procent": model_counts.to_numpy() / len(data) * 100,
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

    subreddit_sizes = data.groupby("subreddit", observed=True).size()
    top_subreddits = (
        subreddit_sizes[subreddit_sizes >= 200]
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    subreddit_rows = data[data["subreddit"].isin(top_subreddits)]
    subreddit_mix = (
        pd.crosstab(subreddit_rows["subreddit"], subreddit_rows["grupa"], normalize="index")
        .reindex(index=top_subreddits, columns=GROUP_ORDER, fill_value=0)
        .mul(100)
    )
    subreddit_table = subreddit_mix.rename(
        columns={
            group: f"{GROUP_LABELS[group]}_procent"
            for group in GROUP_ORDER
        }
    )
    subreddit_table.insert(
        0,
        "Broj_komentari",
        subreddit_sizes.reindex(top_subreddits).astype(int),
    )
    subreddit_table.index.name = "Reddit_zaednica"
    subreddit_table = (
        subreddit_table.sort_values(f"{GROUP_LABELS['negative']}_procent")
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    emotion_path = output_dir / "chestota_emocii.csv"
    group_path = output_dir / "rezime_emocionalni_grupi.csv"
    subreddit_path = output_dir / "rezime_reddit_zaednici.csv"
    emotion_table.to_csv(emotion_path, index=False)
    group_table.to_csv(group_path, index=False)
    subreddit_table.to_csv(subreddit_path, index=False)

    group_rows = "\n".join(
        (
            f"| {row.Emocionalna_grupa} | {int(row.Broj_komentari):,} | "
            f"{row.Broj_komentari / len(data) * 100:.1f}% | "
            f"{row.Mediana_zborovi:.0f} | "
            f"{row.Komentari_so_zbor_od_listata_za_psovki_procent:.1f}% | "
            f"{row.Prosechen_ton:.2f} | {row.Prosechna_subjektivnost:.2f} |"
        )
        for row in group_table.itertuples(index=False)
    )
    emotion_rows = "\n".join(
        f"| {row.Emocija} | {int(row.VerA_ML_komentari):,} | "
        f"{row.VerA_ML_procent:.1f}% |"
        for row in emotion_table.nlargest(5, "VerA_ML_komentari").itertuples(
            index=False
        )
    )
    report_path = output_dir / "izvestaj_opisen.md"
    report_path.write_text(
        f"""# Opisen pregled na komentari

## Podatoci i model

Analizirani se {len(data):,} komentari. VerA ML modelot ja odreduva
dominantnata emocija kako emocijata so najvisok skor od 28 emocii; od nea
se izveduva emocionalnata grupa. Sporedbata so izvornite ocenuvachi opfakja
{len(human_dominant):,} komentari so dostupni oznaki.

## Rezultati po emocionalna grupa

| Grupa | Komentari | Udel | Mediana zborovi | So zbor od listata za psovki | Prosechen ton | Subjektivnost |
|---|---:|---:|---:|---:|---:|---:|
{group_rows}

## Pet najchesti dominantni emocii

| Emocija | Komentari spored VerA ML | Udel |
|---|---:|---:|
{emotion_rows}

## Tolkuvanje

Grafikonite za ton, subjektivnost, dolzhina i zborovi od listata za psovki
gi sumiraat vrednostite vo vlezniot CSV po ML-grupa; tie ne se merki za
uspeh na klasifikatorot i ne pokazhuvaat prichinska povrzanost. Reddit
profilot gi prikazhuva dominantnite ML-grupi za deset zaednici so najmalku
200 komentari. Rezultatite se opisni, a ML-oznakite ne se psiholoshka
procena na avtorite.
""",
        encoding="utf-8",
    )

    top_emotions = emotion_table.assign(
        najvisok_procent=lambda table: table[
            ["VerA_ML_procent", "Izvorni_rateri_procent"]
        ].max(axis=1)
    ).nlargest(10, "najvisok_procent").sort_values("najvisok_procent")
    figure_paths = {}

    fig, axis = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top_emotions))
    height = 0.38
    model_bars = axis.barh(
        y - height / 2,
        top_emotions["VerA_ML_procent"].to_numpy(),
        height,
        label=f"VerA ML (n={len(data):,})",
        color=BLUE,
        zorder=2,
    )
    rater_bars = axis.barh(
        y + height / 2,
        top_emotions["Izvorni_rateri_procent"].to_numpy(),
        height,
        label=f"Izvorni ocenuvachi (n={len(human_dominant):,})",
        color=ORANGE,
        zorder=2,
    )
    axis.set_yticks(y, top_emotions["Emocija"])
    axis.set_xlabel("Udel od komentarite (%)")
    axis.set_title(
        "Dominantni emocii: VerA ML model i ocenuvachi",
        loc="left",
        fontweight="semibold",
    )
    axis.bar_label(model_bars, fmt="%.1f%%", padding=3, fontsize=8)
    axis.bar_label(rater_bars, fmt="%.1f%%", padding=3, fontsize=8)
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)
    axis.set_xlim(0, max(12, top_emotions[
        ["VerA_ML_procent", "Izvorni_rateri_procent"]
    ].to_numpy().max() * 1.18))
    figure_paths["emotion_frequency"] = output_dir / "chestota_emocii.png"
    _save_figure(fig, figure_paths["emotion_frequency"])

    charts = [
        (
            "Komentari_so_zbor_od_listata_za_psovki_procent",
            "Komentari so zbor od listata za psovki (%)",
            "Komentari (%) (opseg 0-20)",
            (0, 20),
            "psovki",
        ),
        (
            "Mediana_zborovi",
            "Tipichna dolzhina na komentar",
            "Mediana na zborovi",
            (0, max(20, group_table["Mediana_zborovi"].max() + 2)),
            "dolzhina",
        ),
        (
            "Prosechen_ton",
            "Prosechen ton na tekstot",
            "Prosechen ton (-1 negativno, 1 pozitivno)",
            (-1, 1),
            "ton",
        ),
        (
            "Prosechna_subjektivnost",
            "Prosechna subjektivnost na tekstot",
            "Rezultat (0 do 1)",
            (0, 1),
            "subjektivnost",
        ),
    ]
    for column, title, xlabel, limits, name in charts:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        positions = np.arange(len(group_table))
        values = group_table[column].to_numpy(dtype=float)
        colors = [GROUP_COLORS[group] for group in GROUP_ORDER if group in set(data["grupa"])]
        for position, value, color in zip(positions, values, colors):
            axis.plot(
                [0, value],
                [position, position],
                color=color,
                linewidth=2.5,
                alpha=0.55,
                zorder=2,
            )
        axis.scatter(values, positions, color=colors, s=58, zorder=3)
        axis.set_yticks(
            positions,
            [
                f"{label}  (n={count:,})"
                for label, count in zip(
                    group_table["Emocionalna_grupa"],
                    group_table["Broj_komentari"],
                )
            ],
        )
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlim(*limits)
        axis.set_ylim(len(group_table) - 0.5, -0.5)
        if name == "psovki":
            axis.set_xticks(np.arange(0, 21, 5))
        elif name == "dolzhina":
            axis.set_xticks(np.arange(0, limits[1] + 1, 5))
        if column == "Prosechen_ton":
            axis.axvline(0, color="#344054", linewidth=1, zorder=1)
        _style_axis(axis)
        for position, value in enumerate(values):
            offset = 6 if value >= 0 else -6
            axis.annotate(
                f"{value:.2f}" if abs(value) < 1 else f"{value:.1f}",
                (value, position),
                xytext=(offset, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
                color=INK,
            )
        figure_paths[name] = output_dir / f"{name}.png"
        _save_figure(fig, figure_paths[name])

    fig, axis = plt.subplots(figsize=(10, 6))
    y = np.arange(len(subreddit_table))
    left = np.zeros(len(subreddit_table))
    for group in GROUP_ORDER:
        values = subreddit_table[f"{GROUP_LABELS[group]}_procent"].to_numpy()
        axis.barh(
            y,
            values,
            left=left,
            label=GROUP_LABELS[group],
            color=GROUP_COLORS[group],
            height=0.72,
            zorder=2,
        )
        left += values
    axis.set_yticks(
        y,
        [
            f"{name}  (n={count:,})"
            for name, count in zip(
                subreddit_table["Reddit_zaednica"],
                subreddit_table["Broj_komentari"],
            )
        ],
    )
    axis.set_xlabel("Komentari po dominantna ML emocionalna grupa (%)")
    axis.set_title(
        "Emocionalen profil na Reddit zaednicite",
        loc="left",
        fontweight="semibold",
    )
    axis.set_xlim(0, 100)
    axis.set_xticks(np.arange(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    axis.legend(
        title="Dominantna grupa",
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    _style_axis(axis)
    axis.set_ylim(len(subreddit_table) - 0.5, -0.5)
    figure_paths["subreddits"] = output_dir / "reddit_zaednici.png"
    _save_figure(fig, figure_paths["subreddits"], rect=(0, 0.1, 1, 1))

    print(f"Sumirani komentari: {len(data):,}")
    print(f"Komentari so oznaki od izvornite ocenuvachi: {len(human_dominant):,}")
    print("Emocionalnite grupi se izvedeni od skorovite na VerA ML modelot.")
    print(f"Tabeli i ednostavni grafikoni: {output_dir}")
    return {
        "emotion_table": emotion_path,
        "group_table": group_path,
        "subreddit_table": subreddit_path,
        "report": report_path,
        **figure_paths,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kreiraj opisni tabeli i grafikoni za emociite vo komentarite."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--emotions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    arguments = parser.parse_args()
    run_simple_analysis(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        emotions_path=arguments.emotions,
        model_path=arguments.model,
    )
