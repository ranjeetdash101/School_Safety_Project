
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "SchoolSafetyEmergencyDataset.csv"
MODEL_PATH = BASE_DIR / "school_safety_model.pkl"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            StandardScaler(),
                            [
                                "temperature_c",
                                "smoke_level",
                                "co_ppm",
                                "gas_level",
                                "sound_db",
                                "crowd_count",
                                "motion_index",
                                "flame_detected",
                                "event_duration_min",
                            ],
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["incident_type"],
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def train_model(data_path: Path = DATA_PATH, output_path: Path = MODEL_PATH):
    df = load_dataset(data_path)
    if "severity_label" not in df.columns:
        raise ValueError("The dataset must include the target column 'severity_label'.")

    feature_cols = [
        col for col in df.columns if col not in {"severity", "severity_label"}
    ]
    X = df[feature_cols]
    y = df["severity_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print("Model trained successfully.")
    print(f"Saved model to: {output_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)


def predict_incident(features: dict) -> str:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model is missing. Run the training command first.")

    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([features])
    return model.predict(sample)[0]


def parse_args():
    parser = argparse.ArgumentParser(description="School Safety Emergency Alert Classification")
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--smoke", type=float)
    parser.add_argument("--co", type=float)
    parser.add_argument("--gas", type=float)
    parser.add_argument("--sound", type=float)
    parser.add_argument("--crowd", type=int)
    parser.add_argument("--motion", type=float)
    parser.add_argument("--flame", type=int, choices=[0, 1])
    parser.add_argument("--duration", type=float)
    parser.add_argument("--incident-type", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        train_model()
        return

    required = [
        "temperature_c",
        "smoke_level",
        "co_ppm",
        "gas_level",
        "sound_db",
        "crowd_count",
        "motion_index",
        "flame_detected",
        "event_duration_min",
        "incident_type",
    ]

    values = {
        "temperature_c": args.temperature,
        "smoke_level": args.smoke,
        "co_ppm": args.co,
        "gas_level": args.gas,
        "sound_db": args.sound,
        "crowd_count": args.crowd,
        "motion_index": args.motion,
        "flame_detected": args.flame,
        "event_duration_min": args.duration,
        "incident_type": args.incident_type,
    }

    missing = [field for field in required if values.get(field) is None]
    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
            + ". Example: python school_safety_project.py predict --temperature 40 --smoke 70 --co 25 --gas 78 --sound 95 --crowd 50 --motion 60 --flame 1 --duration 7 --incident-type \"Heavy Smoke\""
        )

    prediction = predict_incident(values)
    print(f"Predicted emergency severity label: {prediction}")


if __name__ == "__main__":
    main()
