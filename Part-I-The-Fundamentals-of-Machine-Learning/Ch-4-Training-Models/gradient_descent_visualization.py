import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

def generate_linear_data(m, noise=0.1):
    rng = np.random.default_rng(seed=69)
    X = rng.random((m,1),dtype="float32")
    y = 4 + 3* X + noise * rng.standard_normal((m,1),dtype="float32")
    return X, y

def add_bias_term(X):
    bias_column = np.ones_like(X)
    return np.c_[bias_column,X]

def batch_gradient_descent(X, y, learning_rate=0.1, n_epochs=1000):
    """
    Returns: theta (final parameters), theta_path (list of parameters) , cost_history (list of costs per epoch)
    """

    m = len(X)
    rng = np.random.default_rng(seed=69)
    theta = rng.standard_normal((2,1))

    cost_history = []
    theta_path = []

    for epoch in range(n_epochs):
        predictions = X @ theta
        errors = predictions - y
        gradients = 2 / m * X.T @ errors
        theta = theta - learning_rate * gradients
        theta_path.append(theta)
        mse_cost = (1 / m) * (errors ** 2).sum()
        cost_history.append(mse_cost)
    return theta, theta_path, cost_history


if __name__ == "__main__":
    X, y = generate_linear_data(100)
    X = add_bias_term(X)

    learning_rates = np.arange(0.01, 0.7, 0.01)
    cost_histories = []
    
    for eta in learning_rates:
        print(f"learning_rate: {eta}")
        theta, theta_path, cost_history = batch_gradient_descent(X, y, eta)
        print(f"Final theta: {theta.ravel()}")
        print(f"Initial Cost: {cost_history[0]:.4f}")
        print(f"Final Cost: {cost_history[-1]:.4f}")
        cost_histories.append(cost_history)
    
    plt.figure(figsize=(10, 6))
    normalized =  (learning_rates - learning_rates.min()) / (learning_rates.max() - learning_rates.min())
    colors = colormaps['viridis'](normalized)
    labels = [f'eta={eta}' for eta in learning_rates]
    for cost, color, label in zip(cost_histories, colors, labels):
        plt.plot(cost, color=color, linewidth=2, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Cost(MSE)')
    plt.title('Batch Gradient Descent: Cost vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
