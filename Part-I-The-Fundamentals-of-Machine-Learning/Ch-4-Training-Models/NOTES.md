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

**Batch Gradient Descent Code Example**

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

Picks a random instance of training data to compute the gradients.

- irregular cost function that only decreases by average over time
- this makes final solution good but not optimal
- irregular cost function means it bounces out of local minima more frequently than batch gradient descent would

*learning schedule* is the function that determines the learning rate at each iteration. Slowing the rate as the function progresses through the data allows it to achieve a more optimal final solution.

**Stochastic Gradient Descent Code Example**

```python
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))  # randomly initialized model parameters

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = rng.integers(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)  # for SGD, do not divide by m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients
```

And the Scikit-Learn built in class:

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
```

> PolynomialFeatures(degree=d) transforms an array containing n features into an array containing (n + d)! / d!n! features, where n! is the factorial of n, equal to 1 × 2 × 3 × ⋯ × n. 

This seems kind of like a method of data preprocessing for training a linear model like a linear regression? 

### Mini-Batch Gradient Descent

- *mini-batches*: small random sets of instances
- less erratic, closer to minimum, less likely to escape local minima


## Polynomial Regression

Data that does not follow a linear form, such as quadratic functions can be fit with linear models using powers added to features to create a *polynomial regression*.

The following code transforms some data to a second degree polynomial by adding a square of features to the features?

## Learning Curves

Plotting the training and validation error vs the iteration can be helpful in determining a models performance relative to the data, and to see if it is over or underfitting the data.

The Scikit-Learn `learning_curve()` module will train the model using cross-validation and graph and returns the results, sizes, and folds.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
[...]  # beautify the figure: add labels, axis, grid, and legend
plt.show()
```

How to read a Learning Curve:

- both curves plateau indicates an *underfit*, since adding new examples is not going to make the model any better if it underfits the data.
- a gap between the curves indicates an *overfit*, model performs better on the training data than on the validation data

```python
>>> from sklearn.preprocessing import PolynomialFeatures
>>> poly_features = PolynomialFeatures(degree=2, include_bias=False)
>>> X_poly = poly_features.fit_transform(X)
>>> X[0]
array([1.64373629])
>>> X_poly[0]
array([1.64373629, 2.701869  ])
```

## Early Stopping 

Stopping training as soon as validation error reaches a minimum (right before it starts to go up) as this prevent overfitting. Can also be done as a comparison over time for curves that are not smooth such as with Stochastic GD.

## Questions

1. Not sure I really understand the Stochastic Gradient Descent code example? Just in particular the iteration, is that a slice of the training data xi yi ? and then the learning_schedule arguments provided in the iteration loop.. 

2. How exactly do mini-batch and stochastic Gradient descent differ? Specifically what's the difference between an 'iteration' in stochastic GD and a 'min-batch' in mini-batch GD?

3. Why does mini-batch benefit from hardware boosting of matrix multiplications from GPUs .. arean't ALL of these boosted in that way? Why does this method ge extra benefit? Does it involve more matrix operations to get the mini-batches vs a full batch or instance?

4. How would you set up a mini-batch GD if ScikitLearn only has an `SGDRegressor`?

    I guess there's this in the notebook:

    ```python
    theta_path_mgd = []
    for epoch in range(n_epochs):
        shuffled_indices = rng.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for iteration in range(0, n_batches_per_epoch):
            idx = iteration * minibatch_size
            xi = X_b_shuffled[idx : idx + minibatch_size]
            yi = y_shuffled[idx : idx + minibatch_size]
            gradients = 2 / minibatch_size * xi.T @ (xi @ theta - yi)
            eta = learning_schedule(epoch * n_batches_per_epoch + iteration)
            theta = theta - eta * gradients
            theta_path_mgd.append(theta)
    ```

    Could I get it explained in a bit more detail? What is being appended (the theta?) and how is the shuffling and everything working differently to Stochastic or Batch GD?

5. Pretty much everything about the polynomial regression data fit .. cause what

6. Pretty much all of the Regularized Linear models make no sense to me, what is a gradient vector, ridge regression, the math is really going over my head the high level sort of makes sense but the equations do not.

7. What is the difference between these parameters? What scenarios would you use either copy method over the other?

    >"Note that the model is copied using copy.deepcopy(), because it copies both the model’s hyperparameters and the learned parameters. In contrast, sklearn.base.clone() only copies the model’s hyperparameters."

