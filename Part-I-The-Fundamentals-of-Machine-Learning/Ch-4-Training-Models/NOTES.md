# Chapter 4. Training Models

Overview of the math, mostly linear algebra and differential equations concepts regarding training ML models. 

At some point I think I need to go back and review my Linear Algebra, get good at doing some basic vector and matrices math by hand again. 

It's been a while since I've taken linear algebra, diff eq, vector calculus etc.. feeling very rusty but everything is familiar enough for it to be frustrating? 

I could implement some of it with the Python math modules to check my work? 

I need to review:

- Matrix Multiplications
- Transposing and Inverting Matrices
- SVDs Single Value Decompositions
- Normal Equation
- Linear Regressions (MSE cost functions)
- Finding Local and global minima (gradient descent) with calculus
- Partial Derivative 

Some kind of like suggested reading -> examples -> practice problems on paper -> more advanced problems with interactive functions in scikit or pandas etc.. maybe skip the on paper cause computers are better anyways. Maybe progress from filling in the blank to actually writing everything myself/free response questions.

## Linear Regression

- creates an equation of weighted sums for params
- uses cost function to determine weights that minimize error

## Gradient Descent

- *learning rate* hyperparameter is size of steps
- linear regression cost function is convex and continuous (no local minima, only global)

### Batch Gradient Descent

- *partial derivatives* find the slope of cost function for each parameter
- *Gradient vector* of the cost function does this on full training data for every step, very costly
- *epoch* is an iteration over the training set

*Batch Gradient Descent Code Example*

```python
eta = 0.1  # learning rate
n_epochs = 1000
m = len(X_b)  # number of instances

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))  # randomly initialized model parameters

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
```

### Stochastic Gradient Descent



