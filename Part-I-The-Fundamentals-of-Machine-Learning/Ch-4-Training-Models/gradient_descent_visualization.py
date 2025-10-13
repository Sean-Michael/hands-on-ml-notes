import numpy as np

def generate_linear_data(m, noise=0.1):
    rng = np.random.default_rng(seed=69)
    X = rng.random((m,1),dtype="float32")
    y = 4 + 3* X + noise * rng.standard_normal((m,1),dtype="float32")
    return X, y

def add_bias_term(X):
    bias_column = np.ones_like(X)
    return np.c_[bias_column,X]

if __name__ == "__main__":
    X, y = generate_linear_data(100)
    X = add_bias_term(X)
    print(f"X shape: {X.shape}")
    print(f"Y Shape: {y.shape}")
    print(f"First 5 X: {X[:5]}")
    print(f"First 5 y: {y[:5]}")