import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model = joblib.load("models/best_model.joblib")


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("data/train.csv")

target = "survived"

X = df.drop(columns=[target])
y = df[target]


# ============================================================
# 3. SAME TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)


# ============================================================
# 4. PREDICT
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 5. EVALUATE
# ============================================================

print("Evaluation Results")
print("==================")

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))