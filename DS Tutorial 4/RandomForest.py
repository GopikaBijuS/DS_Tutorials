from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class SimpleRandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)  # average predictions

# Example usage
X, y = make_regression(n_samples=100, n_features=5, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf = SimpleRandomForestRegressor(n_estimators=5, max_depth=3)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test, predictions))
