import argparse
import json
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def load_dataset(path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols: List[str] = [
        c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")
    ]
    numeric_cols: List[str] = [c for c in X.columns if c not in categorical_cols]
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                StandardScaler(),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No usable features found in dataset")
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def build_model(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", classifier),
        ]
    )
    return model


def train(
    data_path: str,
    target_column: str = "winner",
    model_path: str = "cricket_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    X, y = load_dataset(data_path, target_column)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(y.unique()) > 1 else None,
    )
    model = build_model(X_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    metrics = {
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "f1_macro": float(f1_score(y_valid, y_pred, average="macro")),
        "classes": list(np.unique(y_valid)),
        "report": classification_report(y_valid, y_pred, output_dict=False),
    }
    joblib.dump({"model": model, "target_column": target_column}, model_path)
    return metrics


def load_trained_model(model_path: str) -> Dict[str, Any]:
    obj = joblib.load(model_path)
    if not isinstance(obj, dict) or "model" not in obj or "target_column" not in obj:
        raise ValueError("Invalid model file")
    return obj


def predict_single(model_obj: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    model: Pipeline = model_obj["model"]
    X_input = pd.DataFrame([features])
    proba = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        classes = model.named_steps["clf"].classes_
        probs = {str(label): float(p) for label, p in zip(classes, proba)}
    else:
        probs = {}
    pred = model.predict(X_input)[0]
    return {
        "prediction": str(pred),
        "probabilities": probs,
    }


def cli_train(args: argparse.Namespace) -> None:
    metrics = train(
        data_path=args.data,
        target_column=args.target,
        model_path=args.model,
        test_size=args.test_size,
    )
    print(json.dumps(metrics, indent=2))


def cli_predict(args: argparse.Namespace) -> None:
    model_obj = load_trained_model(args.model)
    if args.json is not None:
        features = json.loads(args.json)
    else:
        features = {}
        for pair in args.feature or []:
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            if v.replace(".", "", 1).isdigit():
                if "." in v:
                    v_cast: Any = float(v)
                else:
                    v_cast = int(v)
            else:
                v_cast = v
            features[k] = v_cast
    result = predict_single(model_obj, features)
    print(json.dumps(result, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--target", default="winner")
    train_parser.add_argument("--model", default="cricket_model.joblib")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.set_defaults(func=cli_train)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", default="cricket_model.joblib")
    predict_parser.add_argument("--json")
    predict_parser.add_argument(
        "--feature",
        "-f",
        action="append",
    )
    predict_parser.set_defaults(func=cli_predict)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

