# Pandas

This library has lots of helpful things for data analysis, kind of like excel. 

Everything mostly revolves around the `DataFram` which is like a 2D table with columb names, row labels. 

### Setup

Usually people import as `pd`

```python
import pandas as pd
```

### Objects

- `Series` objects is a 1D array, similar to a column in a spreadsheet (with a column name and row labels)
- `DataFrame` objects. This is a 2D table, similar to a spreadsheet. (with column names and row labels)
- `Panel` objects are like dictionaries of `DataFrame`s, less used.

Most methods return modified copies, they do not alter the original object.

### Dataframes

Add with `.assign()` remove with `.pop()`, add in a particular index with `insert()`