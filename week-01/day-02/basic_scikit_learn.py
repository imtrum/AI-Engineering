from re import S
from tarfile import data_filter

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

X = np.array([
    [20, 170],
    [25, 175],
    [30, 180],
    [35, 185],
    [40, 190]
])


# X = np.array([
#     [1],
#     [2],
#     [3],
#     [4],
#     [5],
#     [6],
#     [7],
#     [8],
#     [9],
#     [10]
# ])

# y = np.array([
#     0,
#     0,
#     0,
#     0,
#     0,
#     1,
#     1,
#     1,
#     1,
#     1
# ])

# plt.scatter(X,y)
# plt.xlabel("hours")
# plt.ylabel("success/failure")
# plt.show()

model = LogisticRegression()

X_train, X_test= train_test_split(X, test_size = 0.2 ,random_state = 42)

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled =  scaler.transform(X_test)   

print(X_train_scaled)
print(X_test_scaled)

# model.fit(X_train,y_train)
# predictions = model.predict(X_test)
# f1 = f1_score(y_test,predictions)

# print(predictions)
# print(f1)