import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # Regularization strength
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Ensure labels are -1 or 1
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # No misclassification
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    # Misclassified point
                    dw = self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]

                # Gradient descent updates
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

# Generate 2-class data
X, y = make_blobs(n_samples=200, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)  # SVM needs labels as -1 and 1

# Train
svm = SVM()
svm.fit(X, y)
predictions = svm.predict(X)

# Accuracy
acc = np.mean(predictions == y)
print("Accuracy:", acc)

# Plot
def plot_svm(X, y, model):
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='bwr')

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, model.w, model.b, 0)
    x1_2 = get_hyperplane_value(x0_2, model.w, model.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, model.w, model.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, model.w, model.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, model.w, model.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, model.w, model.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k--')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k--')

    ax.set_xlim([x0_1 - 1, x0_2 + 1])
    ax.set_ylim([min(x1_1, x1_2) - 3, max(x1_1, x1_2) + 3])
    plt.show()

plot_svm(X, y, svm)
