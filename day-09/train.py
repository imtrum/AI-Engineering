import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("src/ai_engineering/data/train.csv")

print("Dataset shape:", df.columns)


# ============================================================
# 2. FEATURES / TARGET
# ============================================================
# print(df.shape) # check if the data successfully loaded
num_features = ["Age","SibSp","Parch","Fare"]
cat_features = ["Pclass","Embarked","Sex"]
target = "Survived"

####### check all the features fine
# features = num_features + cat_features + target

# missing_features = [col for col in features if col not in df.columns]

# if not missing_features:
#     print("OK!")



# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================
X = df[num_features + cat_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# ============================================================
# 4. PREPROCESSING
# ============================================================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("encoder", StandardScaler())
])
cat_pipeline = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                        ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
            ("num",num_pipeline,num_features),
            ("cat",cat_pipeline,cat_features)
])


# ============================================================
# 5. MODELS
# ============================================================

models = {
    "Logistic regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators =  100,
        random_state = 42
    ),
}


# ============================================================
# 5. MODELS
# ============================================================

results = []
trained_models = {}


for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train,y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

# ============================================================
# 7. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv("day-09/results.csv", index = False)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df)

# ============================================================
# 8. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["f1"].idxmax(),
    "model"
]

best_pipeline = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")


# ============================================================
# 9. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_pipeline,
    "models/best_model.joblib"
)

print("Saved: models/best_model.joblib")
