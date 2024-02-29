import numpy as np

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def compute_cost(theta, X, y):
    """Compute the cost (loss) for logistic regression."""
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """Gradient Descent to optimize theta."""
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta -= learning_rate * gradient

        cost = compute_cost(theta, X, y)
        cost_history.append(cost)

    return theta, cost_history

def train_logistic_regression(X, y, num_classes, learning_rate, num_iterations):
    """Train logistic regression models for each class using one-vs-all strategy."""
    theta_models = {}

    for i in range(num_classes):
        # Convert y to binary for the current class
        y_binary = (y == i).astype(int)

        # Initialize theta with zeros
        theta = np.zeros(X.shape[1])

        # Train logistic regression for the current class
        theta, _ = gradient_descent(X, y_binary, theta, learning_rate, num_iterations)

        theta_models[i] = theta

    return theta_models

def predict(X, theta_models):
    """Make predictions using the trained models."""
    probabilities = np.array([sigmoid(X @ theta) for theta in theta_models.values()]).T
    predictions = np.argmax(probabilities, axis=1)
    return predictions

# Example usage
# Assume X_train and y_train are your feature matrix and target variable
# Make sure to add a column of ones to X_train for the bias term

# Hyperparameters
learning_rate = 0.01
num_iterations = 1000
num_classes = 4

# Add a column of ones for the bias term
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# Train logistic regression models
theta_models = train_logistic_regression(X_train_bias, y_train, num_classes, learning_rate, num_iterations)

# Make predictions on test set
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
predictions = predict(X_test_bias, theta_models)

# Evaluate the model
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")

