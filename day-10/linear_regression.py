import numpy as np


# training data
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
], dtype=float)

y = np.array([
    2,
    4,
    6,
    8,
    10
], dtype=float)



#===========================
### USING Gradient Descent
#===========================


# #model paramaters
# w = 0.0
# b = 0.0

# learning_rate = 0.01
# epochs = 1000

# n = len(X)

# for epoch in range(epochs):
#     # Prediction
#     y_pred = w * X[:, 0] + b

#     #error
#     error = y_pred - y

#     #Gradients
#     dw = (2 / n) * np.sum(error * X[:, 0])
#     db = (2 / n) * np.sum(error)

#     w = w - learning_rate * dw
#     b = b - learning_rate * db

# print("w =", w)
# print("b = ", b)

# #Prediction

# x_new = 6
# prediction = w * x_new + b

# print("Prediction:", prediction)


#=================================
### Directly calculate the parameters
#=================================

X_b = np.c_[np.ones((len(X),1)), X]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(theta)

b = theta[0]
w = theta[1]

print("w = ", w)
print("b = ", b)

#Prediction
x_new = 6

x_new_b = np.array([1, x_new])

prediction = x_new_b @ theta
print("Prediction:", prediction)











