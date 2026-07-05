import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Iris.csv")

print(df.head())
print(df.info())

df.drop("Id", axis=1, inplace=True)

encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy :", accuracy)
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, prediction))

print("\nClassification Report\n")
print(classification_report(y_test, prediction))

plt.figure(figsize=(8,5))

plt.scatter(
    y_test,
    prediction,
    color="royalblue",
    s=70,
    alpha=0.8,
    label="Predicted"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted Iris Species")
plt.xlabel("Actual Species")
plt.ylabel("Predicted Species")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.savefig("Actual_vs_Predicted_Iris.png")

plt.show()

cm = confusion_matrix(y_test, prediction)

plt.figure(figsize=(6,5))

plt.imshow(cm, cmap="viridis")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0,1,2], ["Setosa","Versicolor","Virginica"])
plt.yticks([0,1,2], ["Setosa","Versicolor","Virginica"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()

plt.savefig("Confusion_Matrix.png")

plt.show()

plt.figure(figsize=(7,5))

species = ["Setosa","Versicolor","Virginica"]

counts = pd.Series(prediction).value_counts().sort_index()

plt.bar(species, counts,color='Green')

plt.title("Predicted Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.savefig("Prediction_Species_Distribution.png")

plt.show()

plt.figure(figsize=(5,5))

plt.bar(["Accuracy"], [accuracy * 100],color='blue')

plt.ylim(0,110)

plt.ylabel("Percentage")

plt.title("Decision Tree Accuracy")

plt.text(
    0,
    accuracy * 100+2,
    f"{accuracy*100:.2f}%",
    ha="center"
)

plt.savefig("Accuracy.png")

plt.show()