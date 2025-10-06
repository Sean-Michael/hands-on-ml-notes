# Chapter 1. The Machine Learning Landscape

This chapter serves as a high level overview of key concepts, and terminology to know. 

###  What is Machine Learning?

The field of study or practice of 'teaching' a computer program to learn from experience and perform better at tasks. 

The example data a system uses to learn are called the *training set*, each training example is called a *training instance* (or *sample*). The part of an ML system that learns and makes predictions is called a *model*. 

Digging into large amounts of data to discover hidden patterns is called *data mining*. 

## Training Supervision

ML systems categorized according to the amount and type of supervision they get during training: supervised, unsupervised, self-supervised, semi-supervised, and reinforcement learning.

### Supervised learning

The training set fed to the algorithm includes the desired solutions, called *labels*.

A typical task is *classifcation*. Another is to predict a *target* numeric value, such as the prie of a car, given a set of *features* (mileage, age, brand, etc.). This is called *regression*.

### Unsupervised learning

Here the training data is unlabeled, the system tries to learn without a teacher. ex. *clustering*, *visualization*, *dimensionality reduction*, *anomaly detection*, and *association rule learning*

### Semi-supervised learning

These algorithms deal with data that's partially labeled. Most are combinations of unsupervised and supervised algorithms.

### Self-supervised learning

Involves generating a labeled dataset from an unlabled one.

> Transferring knowledge from one task to another is called *transfer learning*

### Reinforcement learning

This involves a learning system, called an *agent* perform actions based on it's strategy, called a *policy* which it tunes based on *rewards* and *penalties*. Like classical conditioning. 

## Batch vs. Online Learning

Some models must be trained upfront on a full dataset -- *batch learning*, while others can be trained one batch at a time, for example, using *gradient descent* -- this is called online learning. 

### Batch learning

Training using all the available data takes a lot of time and compute resources, so is typically done offline. First it's trained, and then it's launched into production and runs without learning anymore, applying what it has learned *offline learning*.

The model will eventually need to be retrained from scratch, a process that can be automated easily at the cost of compute and time. 

### Online learning 

Useful for training systems that need to adapt to change extremely rapidly, on less compute, or on larger models that don't fit in memory by training on portions *out-of-core* learning.

- *learning rate* how fast the model adapts to changing data

A high learning rate will rapidly adapt to new data, but tent to quickly forget the old data: *catastrophic forgetting*.

## Instance-Based Versus Model-Based Learning

The way in which ML systems *generalize* is another point of differentiation: instance-based learning and model-based learning.

### Instance-based learning

- study the data
- select a model
- train the model on training data (like minimizing cost function)
- Finally, applied the model to make predictions on new cases (this is called *inference*), hoping that this model will generalize well.

## Main Challenges of Machine Learning

Since the main task is to select a model and train it on some data, two things can go wrong: "bad model" and "bad data".

### Data Quality 

- *sampling noise* : a sample too small will have this as a result of nonrepresentative data
- *sampling bias* : in large samples if the sampling method is flawed. 

Cleaning data is a significant part of work for professionals. 
- if some are outliers, it may help to discard them
- if some instances are missing a few features, deciding whether to ignore this attribute, instances, or fill in missing values.

A critical part of the success of an ML project is coming up with a good set of features to train on. This process, called *feature engineering*, involves the following steps:

- *Feature selection* (selecting the most useful features to train on among existing features)
- *Feature extraction* (combining existing featuresto produce a more useful one)
- Creating new features by gathering new data

### Overfitting the Training Data

Overgeneralizing is called *overfitting* in machine learning: it performs well on the training data, but it does not generalize well -- it's too complex relative to the amount and noisiness of the training data.

This happens when a model begins to notice patterns in the noise and cannot distinguish between what's real or the result of noise.

This can be mitigated by constraining the model to better fit the data, called *regularization*. This is done by tuning *hyperparameters*. 

## Testing and Validating

Trying out a model in prod is a bad idea. So, split data into two sets: the *training set* and the *test set*. You train on the training set, and test using the test set. The error rate on new cases is called the *generalization error* (or *out-of-sample error*). 

This tells you how well it performs on instances it has never seen before. 

If the training error is low but the generalization error is high, it measn the model is overfitting the training data.

> A general rule f 80/20% training to testing is common.

# Questions

### 1. I don't really understand the last sections on hyperparameter tuning, and data mismatch? Might need to dig a bit deeper. Question?

### *Answer:*

### Hyperparameter Tuning

**Hyperparameters** are parameters of the learning algorithm itself (not the model). Unlike model parameters that are learned during training (like weights in a neural network), hyperparameters are set *before* training begins.

**Key concepts:**
- **Examples of hyperparameters**: learning rate, regularization strength, number of layers in a neural network, number of neighbors in k-NN
- **Why tune them?** The "right" hyperparameter values help prevent overfitting while maintaining good performance
- **The validation set approach**: Since you can't use the test set to tune hyperparameters (that would leak information and give overly optimistic results), you typically:
  1. Split training data into a smaller training set + a **validation set** (or **development set**)
  2. Train multiple models with different hyperparameter values on the reduced training set
  3. Evaluate each on the validation set and select the best-performing hyperparameters
  4. Retrain the final model on the full training set (training + validation combined)
  5. Evaluate once on the test set for the final generalization error

**Holdout validation** is when you simply hold out part of the training set for validation. **Cross-validation** involves splitting the training set into multiple folds and validating on each fold sequentially for more robust hyperparameter selection.

### Data Mismatch

**Data mismatch** occurs when your training data is significantly different from the data your model will encounter in production.

**The problem:**
- High generalization error (bad test set performance) could be due to:
  1. **Overfitting** - model is too complex
  2. **Data mismatch** - training and test data come from different distributions

**Example scenario:**
- You build a flower classification app using flowers from the web (training data)
- Users upload photos from mobile phones (production/test data)
- Mobile photos have different lighting, angles, quality → data mismatch

**Solution - Train-dev set:**
To diagnose whether you have overfitting or data mismatch:
1. Split your training data into: actual training set + **train-dev set**
2. The train-dev set comes from the same distribution as training but isn't used for training
3. Compare errors:
   - If train-dev error is high → **overfitting** (model doesn't generalize even within training distribution)
   - If train-dev error is low but test/validation error is high → **data mismatch** (different distributions)

**Fixing data mismatch:**
- Collect more training data that resembles production data
- Engineer features that are robust across both distributions
- Use data augmentation to simulate production conditions

