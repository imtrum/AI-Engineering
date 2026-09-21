import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("src/ai_engineering/data/train.csv")

# print(df.shape)
# print(df.columns)
# print(df.head())
# print(df.isnull().sum())
print(df.groupby("Sex")["Survived"].mean())
print(
    df.groupby(["Sex", "Pclass"])["Survived"]
      .mean()
)
plt.figure()
df["Survived"].value_counts().plot(kind="bar")
plt.title("trumf")
plt.xlabel("survived")
plt.ylabel("Number of passengers")

plt.figure()
df.groupby("Sex")["Survived"].mean().plot(kind="bar")
plt.title("Survive rate by Gender")
plt.ylabel("Survive rate")
plt.show()