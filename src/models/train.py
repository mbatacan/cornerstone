from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

from src.data.load_data import load_data
from src.features.build_features import add_sepal_area
from src.logging.logger import get_logger


def main():
    logger = get_logger("train_model")
    logger.info("Loading data...")
    df = load_data()
    df = add_sepal_area(df)
    logger.info(
        f"Data loaded successfully: cols {df.columns.tolist()}, shape {df.shape}"
    )
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Training model...")
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    logger.info(f"Test accuracy: {score:.3f}")

    joblib.dump(clf, "models/iris_rf.joblib")
    logger.info("Model saved to models/iris_rf.joblib")


if __name__ == "__main__":
    main()
