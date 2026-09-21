from sklearn.linear_model import LogisticRegression

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import  

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_csv("src/ai_engineering/data/train.csv")

# num_features = df.select_dtypes(include="number").columns.tolist()
# cat_features = df.select_dtypes(exclude="number").columns.tolist()
# target = "Survived"
# cat_features.append("Pclass")
# num_features.remove("Pclass")
# num_features.remove(target)

num_features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]
cat_features = [
    "Pclass",
    "Sex",
    "Embarked"
]
target = "Survived"


print(num_features)
print(cat_features)

num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])


model = LogisticRegression()

model_pipeline = Pipeline([
    ("prepocessor", preprocessor),
    ("model", model)
])

X = df[num_features + cat_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

model_pipeline.fit(X_train,y_train)

y_pred = model_pipeline.predict(X_test)

# print(y_pred)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print(cm)


