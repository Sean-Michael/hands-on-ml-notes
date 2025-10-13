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

def batch_gradient_descent(X, y, learning_rate=0.1, n_epochs=1000,
                            patience=50, min_improvement= 1e-6):
    """
    Returns: theta (final parameters), theta_path (list of parameters) , cost_history (list of costs per epoch)
    
    Stopping Conditions:
    1. Convergence: Cost improvement < min_improvement for patience epochs
    2. Divergence: Cost increases or becomes NaN/inf
    """
    no_improvement_count = 0
    best_cost = float('inf')

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
        if np.isnan(mse_cost) or np.isinf(mse_cost):
            print(f"Divergence detected at epoch {epoch}! Stopping early.")
            break
        if len(cost_history) > 1:
            improvement = cost_history[-2] - cost_history[-1]

            if improvement < min_improvement:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Convergence detected at epoch {epoch}! Stopping early.")
                    break
            else:
                no_improvement_count = 0
    return theta, theta_path, cost_history


if __name__ == "__main__":
    X, y = generate_linear_data(100)
    X = add_bias_term(X)

    learning_rates = np.arange(0.01, .1, 0.01)
    cost_histories = []
    epochs_taken = []
    
    for eta in learning_rates:
        print(f"learning_rate: {eta}")
        theta, theta_path, cost_history = batch_gradient_descent(X, y, eta)
        print(f"Final theta: {theta.ravel()}")
        print(f"Initial Cost: {cost_history[0]:.4f}")
        print(f"Final Cost: {cost_history[-1]:.4f}")
        cost_histories.append(cost_history)
        epochs_taken.append(len(theta_path))
    
    fig, (lrep, mse) = plt.subplots(1,2, figsize=(20,12))

    lrep.plot(learning_rates, epochs_taken, 'o-')
    lrep.set_xlabel('Learning Rates')
    lrep.set_ylabel('Epochs Taken')
    lrep.set_title('Early Stopping: eta vs epochs')
    lrep.grid(True)


    normalized =  (learning_rates - learning_rates.min()) / (learning_rates.max() - learning_rates.min())
    colors = colormaps['viridis'](normalized)
    labels = [f'eta={eta}' for eta in learning_rates]
    for cost, color, label in zip(cost_histories, colors, labels):
        mse.plot(cost, color=color, linewidth=2, label=label)
    mse.set_xlabel('Epoch')
    mse.set_ylabel('Cost(MSE)')
    mse.set_title('Batch Gradient Descent: Cost vs Epoch')
    mse.legend()
    mse.grid(True)
    
    plt.show()
