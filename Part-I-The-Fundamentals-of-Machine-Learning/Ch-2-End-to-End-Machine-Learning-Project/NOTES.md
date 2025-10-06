# Chapter 2. End-to-End Machine Learning Project

This chapter works through a real example of analyzing data for a real estate company. The goal being to illustrate the main steps of a machine learning project. 

### Working with Real Data

The following are open data repositories popular with ML:

- Google Datasets Search
- Hugging Face Datasets
- OpenML.org
- Kaggle.com
- PapersWithCode.com
- UC Irvine Machine Learning Repository
- Stanford Large Network Dataset Collection
- Amazon’s AWS datasets
- U.S. Government’s Open Data
- DataPortals.org
- Wikipedia’s list of machine learning datasets

For our example we're using California Census data on home prices.

### Looking at the Big Picture

Our data includes metrics like population, median income, and median housing price. 

Our goal is to create a model that learns from this data and predicts the median housing price in any distrect given all other metrics.

> Use the machine learning project [checklist](https://github.com/ageron/handson-mlp/blob/main/ml-project-checklist.md)

### Frame the Problem

We want to build a model that replaces manual analysis of metrics to report an accurate estimate of the median home price in a new district. That way other models can use this data to make other predictions. This will replace an error prone method of following many rules to estimate the median price by hand. 

> Pipelines are sequences of data processing components. By breaking up components different steps can run asynchronously and be handled by separate teams, with appropriate monitoring to catch bottlenecks and breakages before they hit downstream components.

I think this is a regression task where we are trying to find a best fit that correlates these metrics with the metric we are trying to predict, median house prices. For this I think that, given our dataset of many median home prices in existing areas it makes sense to use supervised learning. That way we train our model on data that is labelled i.e. *x* metrics correspond with *y* median home price etc. 

*Answer* This is a typical supervised learning task, model can be trained with labeled examples (each instance comes with the expexted output).

It's a typical regression task, asking to predict a value specifically *multiple regression* since it usese multiple features (population, median income, etc.) to make a prediction. Also a *univariate regression* problem since we are trying to predict only a single value for each district, as opposed to a *multivariate regression*.

Finally. since data is remaining the same and won't be changed with new data soon, and is small enough to fit in memory, batch learning should work just fine.

### Creating a test set

We will set aside some mount of dat ~20% at random for later error testing.

```python
import numpy as np

def shuffle_and_split_data(data, test_ration, rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = suffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

This is prone to some error due to random number regeneration. Scikit-Learn provides a similar function `train_test_split()`

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_fill, test_size= 0.2, random_state=42)
```

Random sampling might work for large datasets but runs the risk of sampling bias. 

One way to combat this is *stratified sampling* separating into subgroups called *strata* so that the sample set is representive of the real population/demographic.

For our project it makes sense that median income is a predictor of median housing prices. So we will make sure that we capture a representative test set of the various categories. 

Creating a median income category from the data can be done with `pd.cut()` function from pandas. 

```python
housing_full["income_cat"] = pd.cut(
    housing_full["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5])
```

With these categories created we can do stratified sampling, the easiest way is to get a single split using the `train-test_split()` function with the `stratify` argument. 

```python
strat_train_set, strat_test_set = train_test_split(
    housing_full,
    test_size=0.2,
    stratify=housing_full["income_cat"],
    random_state=42)
```

### Visualizing the data.

We can plot the geographic data latitude and longitde to see the distribution of our data. Setting `alpha` transparency allows us to see the density more clearly.

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()
```

Next we can look at housing prices, the radius `s` represents teh district's population, the color `c` price from a predefined map `cmap` called `jet` which ranges from blue to red (low to high).

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100, label="population",
c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10,7))
plt.show()
```

### Experimenting with Attribute Combinations

Creating combinations of attributes can help give them deeper meaning, by itself the number of rooms in a house or number of households is not very helpful, but knowing the number of rooms per household is. 

```python
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
```

Analyzing the correlation with a matrix:

```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```

> Make sure combined features are not too linearly correlated with existing features: *collinearity* can cause issues with some models like linear regression, particularly avoid simple weighted sums of existing features

## Prepare the Data for Machine Learning Algorithms

Automate the process of preparing the data for the ML algorithms.

### Clean the Data

Missing features will break most algorithms, we noted that `total_bedrooms` has some missing values, we can hangle this in one of three ways:

1. Get rid of the corresponding districts.
2. Get rid of the whole attribute.
3. Set the missing values to some value (zero, the mean, the median, etc.) *imputation*

Pandas DataFrame methods make this easy:

```python
# Drop the districs
housing.dropna(subset=["total_bedrooms"], inplace=True)

# Drop the column
housing.drop("total_bedrooms", axis=1, inplace=True)

# Replace with the median
median = housing["total_bedrooms"].median()
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)
```

Option 3 seems the least destructive. We can use Scikit-Learn's `SimpleImputer` class. 

First create a `SimpleImputer` instance:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
```

> The strategy can also be mean, most_frequent, or constant.

Create a copy of the data with only numerical attributes:

```python
housing_num = housing.select_dtypes(include=[np.number])
```

Fit the `imputer` instance to the training data using the `fit()` method:

```python
imputer.fit(housing_num)
```

This imputer computed the median of each attribute and stored the result in its `statistics_` instance variable. Now we can use it to transform the training set replacing missing values with the medians:

```python
x = imputer.transform(housing_num)
```

Since the Scikit-Learn transformers output NumPy arrays we will wrap `x` in a dataFrame to recover the column names and index from `housing_num`

```python
housing_tr = pd.DataFram(x, coluns=housing_num.columns, index=hosuing_num.index)
```

### Handling Text and Categorical Attributes

The `ocean_proximity` attribute is a limited value category represented by text. Machines prefer numbers so we need to encode these values. 

```python
from skleanr.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
```

