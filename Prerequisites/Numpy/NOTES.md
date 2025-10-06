# NumPy

A fundamental library for scientific computing with Python.

Centered around an *N-dimensional array object*. 

### Creating Arrays

Importing is easy as:

```python
import numpy as np
```

### `np.zeros`

The `zeros` function creates an array of *n* zeroes

```python
> np.zeros(5)
array([0., 0., 0., 0., 0.])
```

A 2-D array (i.e. a matrix) can be created by providing a tuple or rows and columns. 

```python
> np.zeros((3,4))
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
```

### Vocabulary

This is some linear algebra review mostly.

- each dimension is an **axis**
- the number of axes is called the **rank**
    - above 3x4 matris is an array of rank 2 (2-dimesnional)
- list of axis lenghts is called the **shape**
    - aboce matrix's shape is `(3, 4)`
    - The rank is equal to the shape's length
- the **size** is total number of elements, product of all axis lengths (e.g. 3*4=12)

