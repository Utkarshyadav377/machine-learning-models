import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class PerceptronRegressor:
    """Single-hidden-layer perceptron with sigmoid hidden activation and linear output."""

    def __init__(self, hidden_units=16, lr=1e-2, epochs=5000, seed=42):
        self.hidden_units = hidden_units
        self.lr = lr
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _initialize(self, input_dim):
        # Simple Xavier/Glorot initialization
        limit1 = np.sqrt(6 / (input_dim + self.hidden_units))
        self.W1 = self.rng.uniform(-limit1, limit1, size=(input_dim, self.hidden_units))
        self.b1 = np.zeros(self.hidden_units)

        limit2 = np.sqrt(6 / (self.hidden_units + 1))
        self.W2 = self.rng.uniform(-limit2, limit2, size=(self.hidden_units, 1))
        self.b2 = np.zeros(1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n_samples, n_features = X.shape

        if self.W1 is None:
            self._initialize(n_features)

        for _ in range(self.epochs):
            # Forward
            z1 = X @ self.W1 + self.b1
            h1 = sigmoid(z1)
            y_pred = h1 @ self.W2 + self.b2

            # Mean squared error gradients
            error = y_pred - y  # (n,1)
            dW2 = (h1.T @ error) / n_samples
            db2 = np.mean(error, axis=0)

            dh1 = error @ self.W2.T
            dz1 = dh1 * h1 * (1 - h1)  # sigmoid derivative
            dW1 = (X.T @ dz1) / n_samples
            db1 = np.mean(dz1, axis=0)

            # Update
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        z1 = X @ self.W1 + self.b1
        h1 = sigmoid(z1)
        y_pred = h1 @ self.W2 + self.b2
        return y_pred.ravel()


if __name__ == "__main__":
    # Data for y = sin(x). We'll scale targets to [0,1] to pair well with sigmoid.
    rng = np.random.default_rng(0)
    X_train = rng.uniform(-2 * np.pi, 2 * np.pi, size=(256, 1))
    y_train = np.sin(X_train[:, 0])
    y_train_scaled = (y_train + 1) / 2.0

    model = PerceptronRegressor(hidden_units=16, lr=1e-2, epochs=4000, seed=0)
    model.fit(X_train, y_train_scaled)

    # Evaluate
    X_test = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
    y_true = np.sin(X_test[:, 0])
    y_pred_scaled = model.predict(X_test)
    y_pred = 2.0 * y_pred_scaled - 1.0  # invert scaling to get back to [-1,1]

    mse = np.mean((y_pred - y_true) ** 2)
    print("Test MSE:", round(float(mse), 6))

    print("x, sin(x), pred(x):")
    for x, yt, yp in zip(X_test[::25, 0], y_true[::25], y_pred[::25]):
        print(f"{x:+.3f}\t{yt:+.3f}\t{yp:+.3f}")


