import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



numerical_features = ["age","income"]
categorical_features = ["city"]



numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num",numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

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

X = df.drop("bought", axis=1)
y = df["bought"]


scores = cross_val_score(
    model_pipeline,
    X,
    y,
    cv = 5,
    scoring = "accuracy"
)

print(scores)
print(scores.mean())

new_x = pd.DataFrame({
    "age": [30],
    "income": [55000],
    "city": ["Hanoi"]
})

model_pipeline.fit(X,y)
prediction = model_pipeline.predict_proba(new_x)
print(prediction)


# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

# model_pipeline.fit(X_train, y_train)

# predictions = model_pipeline.predict(X_test)

# accuracy = accuracy_score(y_test,predictions)

# print("y_test: ",y_test)

# print(predictions)

# print("accuracy:", accuracy)

# print(model_pipeline)









# X_train_processed = preprocessor.transform(X_train)
# X_test_processed = preprocessor.transform(X_test)

# print(X_test_processed)
# print(X_train_processed)

# df["age"] = df["age"].fillna(df["age"].mean())
# df["income"] = df["income"].fillna(df["income"].mean())
# df["city"] = df["city"].fillna(df["city"].mode()[0])

# city_encoded = encoder.fit_transform(df[["city"]])
# print(encoder.categories_)




# print(df["city"].mode()[0])
# numerical = df.select_dtypes(include="number").columns.size

