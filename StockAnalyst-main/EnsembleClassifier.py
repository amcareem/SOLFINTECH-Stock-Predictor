import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class EnsembleClassifier:

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.gbt = GradientBoostingClassifier(n_estimators=100)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.gbt.fit(X, y)

    def predict(self, X):
        rf_probs = self.rf.predict_proba(X)
        gbt_probs = self.gbt.predict_proba(X)
        averaged_probs = (rf_probs + gbt_probs) / 2
        return np.argmax(averaged_probs, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

# Using the full Iris dataset
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = EnsembleClassifier()
model.fit(X_train, y_train)
print("Accuracy on test data:", model.score(X_test, y_test))