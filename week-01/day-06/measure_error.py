from sklearn.linear_model import LinearRegression
import pandas as pd
from sqlalchemy import select
import ai_engineering.models as models 
from ai_engineering.data.database import db
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

statement = select(models.Customer)
result = db.execute(statement)
customers = result.scalars().all()

data = [
    {
        "id": customer.id,
        "age": customer.age,
        "income": customer.income,
        "city": customer.city,
        "bought": customer.bought
    }
    for customer in customers
]

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

model = LinearRegression()

model_pipeline = Pipeline([
    ("preprocessor",preprocessor), 
    ("model",model)
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

model = model_pipeline.named_steps["model"]

print("weights", model.coef_)
print("bias", model.intercept_)



# mae = mean_absolute_error(y_pred,y_test)
# mse = mean_squared_error(y_pred,y_test)
# r2 = r2_score()

# print("MAE:", mae)
# print("MSE:", mse)
# print("R²:", r2)
