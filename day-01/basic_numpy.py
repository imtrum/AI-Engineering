import pandas as pd
import matplotlib.pyplot as plt
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "score": [50, 52, 55, 60, 65, 70, 78, 85]
}
df = pd.DataFrame(data)

plt.scatter(df["hours"],df["score"])
plt.xlabel("hours")
plt.ylabel("score")

plt.show()