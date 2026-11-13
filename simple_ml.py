"""Train and use a compact GoEmotions multi-label text classifier."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from emotion_labels import EMOTION_LABELS
from ml_reports import (
    plot_cooccurrence,
    plot_emotion_f1,
    plot_precision_recall,
    plot_score_calibration,
    plot_threshold_curve,
    write_academic_report,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "simplified"
DEFAULT_RESULTS_DIR = ROOT / "results" / "ml"
C_VALUES = (1.0, 5.0)
NGRAM_RANGES = ((1, 1), (1, 2))
THRESHOLDS = tuple(round(value / 1000, 3) for value in range(50, 501, 25))
RANDOM_SEED = 42


@dataclass(frozen=True)
class DatasetSplit:
    texts: list[str]
    labels: np.ndarray
    ids: list[str]


@dataclass(frozen=True)
class CandidateResult:
    model: Pipeline
    C: float
    ngram_range: tuple[int, int]
    threshold: float
    dev_macro_f1: float
    threshold_scores: tuple[tuple[float, float], ...]


def load_emotions(data_dir: Path) -> list[str]:
    path = data_dir / "emotions.txt"
    emotions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    emotions = [emotion for emotion in emotions if emotion]
    if not emotions or len(emotions) != len(set(emotions)):
        raise ValueError(f"Emotion list must be non-empty and unique: {path}")
    return emotions


def load_split(path: Path, emotions: Sequence[str]) -> DatasetSplit:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "emotion_ids", "id"],
        dtype="string",
        keep_default_na=False,
    )
    if frame.empty:
        raise ValueError(f"Dataset split is empty: {path}")
    if frame[["text", "id"]].apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError(f"Dataset split has empty text or IDs: {path}")
    if frame["id"].duplicated().any():
        raise ValueError(f"Dataset split has duplicate IDs: {path}")

    labels = np.zeros((len(frame), len(emotions)), dtype=np.int8)
    for row_index, raw_ids in enumerate(frame["emotion_ids"]):
        if not raw_ids.strip():
            continue
        try:
            emotion_ids = {int(value.strip()) for value in raw_ids.split(",")}
        except ValueError as exc:
            raise ValueError(
                f"Invalid emotion IDs in {path}, row {row_index + 1}: {raw_ids!r}"
            ) from exc
        if any(index < 0 or index >= len(emotions) for index in emotion_ids):
            raise ValueError(
                f"Emotion ID is out of range in {path}, row {row_index + 1}"
            )
        labels[row_index, list(emotion_ids)] = 1
    return DatasetSplit(
        texts=frame["text"].astype(str).tolist(),
        labels=labels,
        ids=frame["id"].astype(str).tolist(),
    )


def _validate_split_ids(splits: dict[str, DatasetSplit]) -> None:
    names = list(splits)
    for index, first_name in enumerate(names):
        first_ids = set(splits[first_name].ids)
        for second_name in names[index + 1 :]:
            overlap = first_ids.intersection(splits[second_name].ids)
            if overlap:
                raise ValueError(
                    f"{first_name} and {second_name} share {len(overlap)} comment IDs"
                )


def build_model(C: float, ngram_range: tuple[int, int]) -> Pipeline:
    if C <= 0:
        raise ValueError("C must be positive")
    if len(ngram_range) != 2 or not 1 <= ngram_range[0] <= ngram_range[1]:
        raise ValueError(f"Invalid ngram range: {ngram_range}")
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    min_df=2,
                    ngram_range=ngram_range,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=C,
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    )
                ),
            ),
        ]
    )


def _macro_f1_by_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> list[tuple[float, float]]:
    return [
        (
            threshold,
            float(
                f1_score(
                    labels,
                    scores >= threshold,
                    average="macro",
                    zero_division=0,
                )
            ),
        )
        for threshold in THRESHOLDS
    ]


def _is_better_candidate(
    candidate: CandidateResult,
    current: CandidateResult | None,
) -> bool:
    if current is None or candidate.dev_macro_f1 != current.dev_macro_f1:
        return current is None or candidate.dev_macro_f1 > current.dev_macro_f1
    candidate_complexity = (candidate.ngram_range[1], candidate.C)
    current_complexity = (current.ngram_range[1], current.C)
    return candidate_complexity < current_complexity


def _evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    emotions: Sequence[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("Evaluation labels and scores must have matching matrices")
    if labels.shape[1] != len(emotions):
        raise ValueError("Evaluation columns do not match the emotion list")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model scores must be finite values in [0, 1]")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
    predictions = scores >= threshold
    macro = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    micro = precision_recall_fscore_support(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        zero_division=0,
    )
    summary = {
        "Makro_preciznost": float(macro[0]),
        "Makro_odziv": float(macro[1]),
        "Makro_F1": float(macro[2]),
        "Mikro_preciznost": float(micro[0]),
        "Mikro_odziv": float(micro[1]),
        "Mikro_F1": float(micro[2]),
        "Tocnost_na_celiot_zapis": float(accuracy_score(labels, predictions)),
        "Pogreshna_oznaka_del": float(hamming_loss(labels, predictions)),
        "Prosechno_oznaki_po_komentar": float(labels.sum(axis=1).mean()),
    }
    per_emotion = pd.DataFrame(
        {
            "Emocija": [EMOTION_LABELS.get(emotion, emotion) for emotion in emotions],
            "Preciznost": precision,
            "Odziv": recall,
            "F1": f1,
            "Poddrshka": support.astype(int),
        }
    )
    return summary, per_emotion


def train_model(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, object]:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    emotions = load_emotions(data_dir)
    splits = {
        name: load_split(data_dir / f"{name}.tsv", emotions)
        for name in ("train", "dev", "test")
    }
    _validate_split_ids(splits)
    train_split = splits["train"]
    dev_split = splits["dev"]
    test_split = splits["test"]
    train_support = train_split.labels.sum(axis=0)
    unsupported = [
        emotion
        for emotion, support in zip(emotions, train_support)
        if support == 0 or support == len(train_split.texts)
    ]
    if unsupported:
        raise ValueError(
            "Training split must contain positive and negative examples for "
            f"each emotion: {unsupported}"
        )

    best: CandidateResult | None = None
    candidate_rows = []
    for C in C_VALUES:
        for ngram_range in NGRAM_RANGES:
            model = build_model(C, ngram_range)
            model.fit(train_split.texts, train_split.labels)
            dev_scores = model.predict_proba(dev_split.texts)
            curve = _macro_f1_by_threshold(dev_split.labels, dev_scores)
            threshold, dev_macro_f1 = max(curve, key=lambda result: result[1])
            candidate = CandidateResult(
                model=model,
                C=C,
                ngram_range=ngram_range,
                threshold=threshold,
                dev_macro_f1=dev_macro_f1,
                threshold_scores=tuple(curve),
            )
            candidate_rows.append(
                {
                    "C": C,
                    "Ngram": f"{ngram_range[0]}-{ngram_range[1]}",
                    "Prag": threshold,
                    "Dev_makro_F1": dev_macro_f1,
                }
            )
            if _is_better_candidate(candidate, best):
                best = candidate

    if best is None:
        raise RuntimeError("Model tuning produced no candidate")
    test_scores = best.model.predict_proba(test_split.texts)
    test_summary, per_emotion = _evaluate(
        test_split.labels,
        test_scores,
        best.threshold,
        emotions,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows).sort_values(
        "Dev_makro_F1", ascending=False
    ).to_csv(results_dir / "dev_model_selection.csv", index=False)
    pd.DataFrame(
        [
            {
                "Metrika": metric,
                "Vrednost": value,
            }
            for metric, value in test_summary.items()
        ]
    ).to_csv(results_dir / "test_metrici.csv", index=False)
    per_emotion.to_csv(results_dir / "emocionalni_metrici.csv", index=False)
    pd.DataFrame(
        [
            {"Prag": value, "Dev_makro_F1": score}
            for value, score in best.threshold_scores
        ]
    ).to_csv(results_dir / "pragovi_makro_f1.csv", index=False)
    plot_threshold_curve(best.threshold_scores, best.threshold, results_dir)
    plot_emotion_f1(per_emotion, results_dir)
    plot_precision_recall(per_emotion, results_dir)
    plot_cooccurrence(
        test_split.labels,
        emotions,
        train_support,
        results_dir,
    )
    plot_score_calibration(
        test_split.labels,
        test_scores,
        emotions,
        train_support,
        results_dir,
    )

    artifact = {
        "model": best.model,
        "emotions": emotions,
        "threshold": best.threshold,
    }
    joblib.dump(artifact, results_dir / "model.joblib")
    selected = {
        "algorithm": "TF-IDF + One-vs-Rest Logistic Regression",
        "C": best.C,
        "ngram_range": best.ngram_range,
        "threshold": best.threshold,
        "development_macro_f1": best.dev_macro_f1,
        "train_comments": len(train_split.texts),
        "development_comments": len(dev_split.texts),
        "test_comments": len(test_split.texts),
        "feature_count": len(best.model.named_steps["tfidf"].vocabulary_),
        "random_seed": RANDOM_SEED,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    (results_dir / "model_info.json").write_text(
        json.dumps(selected, indent=2),
        encoding="utf-8",
    )
    write_academic_report(selected, test_summary, per_emotion, results_dir)
    print(f"Komentari za obuka: {len(train_split.texts):,}")
    print(f"Makro-F1 na razvojniot skup: {best.dev_macro_f1:.3f}")
    print(f"Makro-F1 na test-skupot: {test_summary['Makro_F1']:.3f}")
    print(f"Izbran prag: {best.threshold:.3f}")
    print(f"Rezultati i model: {results_dir}")
    return {
        "model": best.model,
        "threshold": best.threshold,
        "development_macro_f1": best.dev_macro_f1,
        "test_metrics": test_summary,
        "per_emotion_metrics": per_emotion,
        "results_dir": results_dir,
    }


def predict_text(text: str, model_path: str | Path) -> list[tuple[str, float]]:
    if not text.strip():
        raise ValueError("Tekstot za predviduvanje ne smee da bide prazen")
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or not {
        "model",
        "emotions",
        "threshold",
    } <= set(artifact):
        raise ValueError(f"Nevalidna datoteka so model: {model_path}")
    model = artifact["model"]
    emotions = artifact["emotions"]
    threshold = artifact["threshold"]
    if (
        not isinstance(emotions, list)
        or not all(isinstance(emotion, str) for emotion in emotions)
        or not isinstance(threshold, (int, float))
        or not 0 < threshold < 1
    ):
        raise ValueError(f"Invalid model metadata: {model_path}")
    scores = model.predict_proba([text])[0]
    if scores.shape != (len(emotions),):
        raise ValueError(f"Model output does not match its emotion list: {model_path}")
    labels = [
        (EMOTION_LABELS.get(emotion, emotion), float(scores[index]))
        for index, emotion in enumerate(emotions)
        if scores[index] >= threshold
    ]
    return sorted(labels, key=lambda result: result[1], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ednostaven GoEmotions model so TF-IDF i logistichka regresija."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Obuchi, izberi prag i proceni model")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    predict = commands.add_parser("predict", help="Predvidi emocii za eden tekst")
    predict.add_argument("--text", required=True)
    predict.add_argument("--model", type=Path, default=DEFAULT_RESULTS_DIR / "model.joblib")
    arguments = parser.parse_args()

    if arguments.command == "train":
        train_model(arguments.data_dir, arguments.results_dir)
        return
    labels = predict_text(arguments.text, arguments.model)
    if not labels:
        print("Nema emocija nad izbraniot prag.")
        return
    for emotion, score in labels:
        print(f"{emotion}: {score:.3f}")


if __name__ == "__main__":
    main()
