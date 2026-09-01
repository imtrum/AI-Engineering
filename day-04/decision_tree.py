import random

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


data = {
"age": [
    20, 25, None, 35, 40,
    22, 28, 31, None, 45,
    50, 27, 33, 38, 42,
    None, 24, 29, 36, 48
],

"income": [
    30000, 40000, 50000, None, 80000,
    35000, 45000, 52000, 60000, None,
    90000, 42000, 55000, 65000, 75000,
    48000, None, 47000, 62000, 85000
],

"city": [
    "Hanoi", "Hanoi", "Da Nang", "Hanoi", None,
    "Ho Chi Minh", "Hanoi", "Da Nang", "Ho Chi Minh", "Hanoi",
    "Da Nang", None, "Hanoi", "Ho Chi Minh", "Da Nang",
    "Hanoi", "Ho Chi Minh", "Da Nang", "Hanoi", "Ho Chi Minh"
],

"bought": [
    0, 1, 1, 0, 1,
    0, 1, 1, 1, 0,
    1, 0, 1, 1, 1,
    0, 0, 1, 0, 1
]
}

df = pd.DataFrame(data)
X = df.drop("bought",axis=1)
y = df["bought"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

numerical_features = ["age","income"]
categorical_features = ["city"]

numerical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="mean")),
    ("Scaler",StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ("nums",numerical_pipeline,numerical_features),
    ("cat",categorical_pipeline,categorical_features)
])

param_grid = {
    "model__max_depth":[1,2,3,4,5,10]
}

tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier())
])

grid_search = GridSearchCV(
    tree_pipeline,
    param_grid,
    cv = 5,
    scoring="accuracy"
)

grid_search.fit(X,y)

print(grid_search.best_estimator_)
print(grid_search.best_score_)



# tree_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", DecisionTreeClassifier(
#             max_depth=2,
#             random_state=42
#     ))
# ])

# logistic_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", LogisticRegression())
# ])


# logistic_scores = cross_val_score(
#     tree_pipeline,
#     X,
#     y,
#     cv = 5,
#     scoring="accuracy"
# )

# tree_scores = cross_val_score(
#     logistic_pipeline,
#     X,
#     y,
#     cv=5,
#     scoring="accuracy"
# )

# print("Logistic Regression:", logistic_scores.mean())
# print("Decision Tree:", tree_scores.mean())

