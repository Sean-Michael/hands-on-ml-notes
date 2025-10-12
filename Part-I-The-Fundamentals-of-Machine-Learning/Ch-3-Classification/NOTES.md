# Chapter 3. Classification

## Confusion Matrices

Cross-Validation does not works so good for classifiers, especially with skewed data sets

The general idea of a confusion matrix is to count the number of times instances of class A are classified as class B, for all A/B pairs. For example, to know the number of times the classifier confused images of 8s with 0s, you would look at row #8, column #0 of the confusion matrix.

To compute the F1 score, simply call the f1_score() function:

>>> from sklearn.metrics import f1_score
>>> f1_score(y_train_5, y_train_pred)
0.7325171197343846
The F1 score favors classifiers that have similar precision and recall. This is not always what you want: in some contexts you mostly care about precision, and in other contexts you really care about recall. For example, if you trained a classifier to detect videos that are safe for kids, you would probably prefer a classifier that rejects many good videos (low recall) but keeps only safe ones (high precision), rather than a classifier that has a much higher recall but lets a few really bad videos show up in your product (in such cases, you may even want to add a human pipeline to check the classifier’s video selection). On the other hand, suppose you train a classifier to detect shoplifters in surveillance images: it is probably fine if your classifier only has 30% precision as long as it has 99% recall. Sure, the security guards will get a few false alerts, but almost all shoplifters will get caught. Similarly, medical diagnosis usually requires a high recall to avoid missing anything important. False positives can be ruled out by follow-up medical tests.

Unfortunately, you can’t have it both ways: increasing precision reduces recall, and vice versa. This is called the precision/recall trade-off.

## Questions

1. What does the random_state do here? `sgd_clf = SGDClassifier(random_state=42)`

*Answer*: The `random_state` parameter sets the seed for the random number generator used during training. SGDClassifier uses randomness when shuffling the training data and initializing weights. Setting `random_state=42` ensures reproducibility—you'll get the same results every time you run the code with the same data. Without it, results would vary slightly between runs.

2. I still am a bit confused by this syntax? It seems super simple/abstracted? what's it actually doing? 

```python
y_train_5 = (y_train == '5')  # True for all 5s, False for all other digits
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

*Answer*: This creates a binary classifier for detecting 5s. The first two lines create boolean arrays where `True` indicates the digit is a 5 and `False` for everything else—this converts the multi-class problem (digits 0-9) into a binary classification problem (5 vs not-5). The `SGDClassifier` then trains using Stochastic Gradient Descent to find a decision boundary that separates 5s from non-5s. The `.fit()` method iterates through the training data, making small weight adjustments to minimize classification error.

3. How does scikit know to use OVR or OVO like how does it know that there's 10 classes just from this?

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])  # y_train, not y_train_5
```

I guess the training data is labeled right so it understands there are 10 labels?

*Answer*: Yes, exactly right! The classifier detects the number of unique classes by examining the `y_train` labels. When you pass `y_train` (which contains digits 0-9), scikit-learn automatically identifies there are 10 distinct classes. For `SVC` specifically, it uses OVO (One-vs-One) by default, creating 45 binary classifiers (one for each pair of digits). If you used a different classifier like `SGDClassifier` with multi-class data, it would use OVR (One-vs-Rest) by default instead, creating 10 binary classifiers (one per digit vs all others).