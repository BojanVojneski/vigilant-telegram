"""Charts and a compact academic-style report for the VerA model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from emotion_labels import EMOTION_LABELS
from plot_style import (
    BLUE,
    GRID,
    INK,
    MUTED,
    NAVY,
    ORANGE,
    RED,
    TEAL,
    apply_plot_style,
)

apply_plot_style()


def _label(emotion: str) -> str:
    return EMOTION_LABELS.get(emotion, emotion)


def plot_threshold_curve(
    scores: Sequence[tuple[float, float, float, float]],
    selected_threshold: float,
    output_dir: Path,
) -> Path:
    curve = pd.DataFrame(
        scores,
        columns=["threshold", "macro_precision", "macro_recall", "macro_f0_5"],
    )
    selected = curve.loc[np.isclose(curve["threshold"], selected_threshold)]
    if selected.empty:
        raise ValueError("Selected threshold is missing from the tuning curve")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        curve["threshold"],
        curve["macro_precision"],
        marker="o",
        color=BLUE,
        label="Makro-preciznost",
    )
    axis.plot(
        curve["threshold"],
        curve["macro_recall"],
        marker="o",
        color=ORANGE,
        label="Makro-odziv",
    )
    axis.plot(
        curve["threshold"],
        curve["macro_f0_5"],
        marker="o",
        color=TEAL,
        label="Makro-F0.5",
    )
    axis.scatter(
        [selected_threshold],
        [selected["macro_f0_5"].iloc[0]],
        marker="*",
        s=140,
        color=RED,
        zorder=3,
        label=f"Izbran prag: {selected_threshold:.3f}",
    )
    axis.axvline(selected_threshold, color=MUTED, linestyle="--", linewidth=1)
    axis.set_title("Preciznost i odziv po prag", loc="left", fontweight="semibold")
    axis.set_xlabel("Prag")
    axis.set_ylabel("Merka na razvojnite podatoci")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = output_dir / "preciznost_odziv_po_pragu.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_emotion_f1(metrics: pd.DataFrame, output_dir: Path) -> Path:
    required = {"Emocija", "F1", "Poddrshka"}
    if not required <= set(metrics.columns) or metrics.empty:
        raise ValueError("Per-emotion metrics are incomplete")
    if not metrics["F1"].between(0, 1).all() or (metrics["Poddrshka"] < 0).any():
        raise ValueError("F1 and support values are invalid")

    ordered = metrics.sort_values(["F1", "Poddrshka"])
    labels = [
        f"{emotion} (n={support})"
        for emotion, support in zip(ordered["Emocija"], ordered["Poddrshka"])
    ]
    fig, axis = plt.subplots(figsize=(9, 9))
    bars = axis.barh(labels, ordered["F1"], color=BLUE)
    axis.set_title("F1 po emocija na test-skupot")
    axis.set_xlabel("F1")
    axis.set_ylabel("Emocija (broj na pozitivni oznaki)")
    axis.set_xlim(0, 1)
    axis.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, ordered["F1"]):
        axis.text(
            min(float(value) + 0.01, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
        )
    fig.tight_layout()
    path = output_dir / "f1_po_emocija.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_precision_recall(metrics: pd.DataFrame, output_dir: Path) -> Path:
    required = {"Emocija", "Preciznost", "Odziv", "Poddrshka"}
    if not required <= set(metrics.columns) or metrics.empty:
        raise ValueError("Precision-recall metrics are incomplete")
    ordered = metrics.sort_values(["Preciznost", "Poddrshka"])
    y = np.arange(len(ordered))
    precision = ordered["Preciznost"].to_numpy()
    recall = ordered["Odziv"].to_numpy()
    fig, axis = plt.subplots(figsize=(9, 10))
    for position, p_value, r_value in zip(y, precision, recall):
        axis.plot(
            [p_value, r_value],
            [position, position],
            color=GRID,
            linewidth=1.5,
            zorder=1,
        )
    axis.scatter(precision, y - 0.08, color=BLUE, s=24, label="Preciznost", zorder=2)
    axis.scatter(recall, y + 0.08, color=ORANGE, s=24, label="Odziv", zorder=2)
    axis.set_yticks(
        y,
        [
            f"{emotion} (n={support})"
            for emotion, support in zip(ordered["Emocija"], ordered["Poddrshka"])
        ],
    )
    axis.set_title(
        "Preciznost i odziv za site emocii",
        loc="left",
        fontweight="semibold",
    )
    axis.set_xlabel("Rezultat")
    axis.set_xlim(0, 1)
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = output_dir / "preciznost_i_odziv.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_cooccurrence(
    labels: np.ndarray,
    emotions: Sequence[str],
    training_support: np.ndarray,
    output_dir: Path,
    top_n: int = 12,
) -> Path:
    if labels.ndim != 2 or labels.shape[1] != len(emotions):
        raise ValueError("Label matrix does not match the emotion list")
    if training_support.shape != (len(emotions),):
        raise ValueError("Training support does not match the emotion list")
    if not 1 <= top_n <= len(emotions):
        raise ValueError("top_n must be between 1 and the number of emotions")
    selected_indices = np.argsort(training_support)[-top_n:][::-1]
    selected_labels = labels[:, selected_indices].astype(np.int64)
    pair_counts = selected_labels.T @ selected_labels
    supports = pair_counts.diagonal()
    conditional = np.divide(
        pair_counts,
        supports[:, np.newaxis],
        out=np.zeros_like(pair_counts, dtype=float),
        where=supports[:, np.newaxis] > 0,
    ) * 100
    names = [_label(emotions[index]) for index in selected_indices]
    diagonal = np.eye(top_n, dtype=bool)
    displayed = np.ma.masked_where(diagonal, conditional)
    cmap = LinearSegmentedColormap.from_list(
        "vera_navy",
        ["#F5F7FA", "#B8CEE0", "#5E8FB6", BLUE, NAVY],
    )
    cmap.set_bad("#E8EDF3")

    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(displayed, cmap=cmap, vmin=0, vmax=20)
    axis.set_xticks(np.arange(top_n), names, rotation=45, ha="right")
    axis.set_yticks(np.arange(top_n), names)
    axis.set_title(
        "Emocii koi se pojavuvaat zaedno",
        loc="left",
        fontweight="semibold",
        pad=18,
    )
    axis.text(
        0,
        1.01,
        "Test-skup | 12 najchesti emocii | dijagonalata e iskluchena",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    axis.set_xlabel("Emocija vo komentarot (kolona)")
    axis.set_ylabel("Uslovna emocija (red)")
    axis.set_xticks(np.arange(-0.5, top_n, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, top_n, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(axis="both", labelsize=9)
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for row in range(top_n):
        for column in range(top_n):
            if diagonal[row, column]:
                axis.text(
                    column,
                    row,
                    "-",
                    ha="center",
                    va="center",
                    color=MUTED,
                    fontsize=9,
                )
                continue
            value = conditional[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color="white" if value >= 12 else INK,
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(
        "Udel so emocija od kolonata, megju komentarite so emocija od redot (%)"
    )
    colorbar.set_ticks(
        [0, 5, 10, 15, 20],
        labels=["0%", "5%", "10%", "15%", "20%"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "ko_pojava_emocii.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_score_calibration(
    labels: np.ndarray,
    scores: np.ndarray,
    emotions: Sequence[str],
    training_support: np.ndarray,
    output_dir: Path,
    top_n: int = 5,
    bins: int = 10,
) -> Path:
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("Calibration labels and scores must have matching matrices")
    if labels.shape[1] != len(emotions) or training_support.shape != (len(emotions),):
        raise ValueError("Calibration data does not match the emotion list")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Calibration scores must be finite values in [0, 1]")
    if not 1 <= top_n <= len(emotions) or bins < 2:
        raise ValueError("Invalid calibration graph settings")
    selected_indices = np.argsort(training_support)[-top_n:][::-1]

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Idealna kalibracija")
    for index in selected_indices:
        order = np.argsort(scores[:, index], kind="stable")
        chunks = np.array_split(order, min(bins, len(order)))
        mean_scores = [scores[chunk, index].mean() for chunk in chunks if len(chunk)]
        observed_rates = [labels[chunk, index].mean() for chunk in chunks if len(chunk)]
        axis.plot(
            mean_scores,
            observed_rates,
            marker="o",
            markersize=3,
            label=_label(emotions[index]),
        )
    axis.set_title("Kalibracija na model-skorovite na test-skupot")
    axis.set_xlabel("Prosechen model-skor")
    axis.set_ylabel("Procent pozitivni oznaki")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)
    axis.legend(title="Emocija", fontsize=8)
    fig.tight_layout()
    path = output_dir / "kalibracija_skorovi.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_academic_report(
    model_info: dict[str, object],
    metrics: dict[str, float],
    per_emotion: pd.DataFrame,
    output_dir: Path,
) -> Path:
    best = per_emotion.nlargest(5, "F1")
    weakest = per_emotion.nsmallest(5, "F1")
    best_rows = "\n".join(
        f"| {row.Emocija} | {int(row.Poddrshka)} | {row.F1:.3f} |"
        for row in best.itertuples(index=False)
    )
    weakest_rows = "\n".join(
        f"| {row.Emocija} | {int(row.Poddrshka)} | {row.F1:.3f} |"
        for row in weakest.itertuples(index=False)
    )
    report = f"""# VerA: izvestaj za klasifikacija na emocii

## Metod

Zadachata e povekje-oznacna klasifikacija na tekst od GoEmotions. Modelot
koristi TF-IDF zborovni n-grami i logistichka regresija so one-vs-rest
klasifikacija. Izborot na C, n-grami i zaednichki prag e napraven samo na
trenirachkite i razvojnite podatoci. Pragot se izbira spored najvisok prosechen
F0.5 po emocija na razvojnata podelba; test-podelbata ne se koristi za izbor.

| Stavka | Vrednost |
|---|---:|
| C | {model_info["C"]} |
| N-gram opseg | {model_info["ngram_range"]} |
| Izbran prag | {model_info["threshold"]:.3f} |
| Izbor na prag | Makro-F0.5 na razvojni podatoci |
| Trenirachki komentari | {model_info["train_comments"]:,} |
| Razvojni komentari | {model_info["development_comments"]:,} |
| Test komentari | {model_info["test_comments"]:,} |
| Razvoen makro-F0.5 | {model_info["development_macro_f0_5"]:.3f} |
| Razvoena makro-preciznost | {model_info["development_threshold_metrics"]["macro_precision"]:.3f} |
| Razvoen makro-odziv | {model_info["development_threshold_metrics"]["macro_recall"]:.3f} |

## Rezultati na test-podelbata

| Merka | Vrednost |
|---|---:|
| Makro-preciznost | {metrics["Makro_preciznost"]:.3f} |
| Makro-odziv | {metrics["Makro_odziv"]:.3f} |
| Makro-F1 | {metrics["Makro_F1"]:.3f} |
| Makro-F0.5 | {metrics["Makro_F0_5"]:.3f} |
| Mikro-F1 | {metrics["Mikro_F1"]:.3f} |
| Tochno poklopeni skupovi oznaki | {metrics["Tocnost_na_celiot_zapis"]:.3f} |

Makro-F1 ja presmetuva prosechnata F1 vrednost so ednakva tezhina za sekoja
emocija. Pragot se bira spored prosechniot F0.5 po emocija, koj dava pogolema
tezhina na preciznosta od odzivot; toa e nameren kompromis i obichno znachi
pomalku predvideni oznaki i pomal odziv. Mikro-F1 gi sobira odlukite za site
emocii pred presmetka.
Tochnoto poklopuvanje bara modelot da gi predvidi site oznaki na komentarot
bez greska, pa ovaa merka e postroga.

## Najvisoki i najniski F1 po emocija

| Emocija | Pozitivni oznaki | F1 |
|---|---:|---:|
{best_rows}

| Emocija | Pozitivni oznaki | F1 |
|---|---:|---:|
{weakest_rows}

## Grafichki prikazi

- `preciznost_odziv_po_pragu.png`: kompromis megju preciznost, odziv i makro-F0.5 na razvojniot skup.
- `f1_po_emocija.png`: test-F1 i brojot na pozitivni primeri za sekoja emocija.
- `preciznost_i_odziv.png`: sporedba na preciznost i odziv za site 28 emocii na test-skupot.
- `ko_pojava_emocii.png`: procentot so emocija od kolonata megju komentarite so emocija od redot; gi prikazhuva 12te najchesti trenirachki emocii, bez trivijalnata dijagonala.
- `kalibracija_skorovi.png`: sporedba na model-skorot so empiriskiot del na pozitivni oznaki za pette najchesti emocii.

## Tolkuvanje i ogranicuvanja

Rezultatite ja merat soglasnosta so oznakite vo ovaa podelba na GoEmotions,
ne vistinskite chuvstva na avtorite. Oznakite se subjektivni i nekoi emocii
imaat mal test-primerok, pa nivnite F1 vrednosti se ponepostojani. Modelot
e ednostavna pochetna osnova, ne psiholoshki alat. Grafikonot za kalibracija
e dijagnostika; skorovite ne se koregirani ili potvrdeni kako apsolutni
verojatnosti.

Ne se koristeni hipotezi, p-vrednosti ili prichinski tvrdenja.
"""
    path = output_dir / "izvestaj_ml.md"
    path.write_text(report, encoding="utf-8")
    return path
