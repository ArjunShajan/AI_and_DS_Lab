from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print("Target names (classes):", cancer.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

plt.figure(figsize=(20, 15))
plot_tree(clf,
          filled=True,
          feature_names=cancer.feature_names,
          class_names=cancer.target_names,
          rounded=True,
          fontsize=12)
plt.title("Decision Tree for Breast Cancer Dataset (Entropy)", fontsize=16)
plt.tight_layout()
plt.show()