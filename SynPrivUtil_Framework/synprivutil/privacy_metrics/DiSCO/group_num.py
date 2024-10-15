import pandas as pd
import numpy as np


# def group_num(x1, x2, xsyn, n=5, style="quantile", cont_na=None, **kwargs):
#     """
#     Categorizes two continuous variables into factors of n groups with the same groupings determined by the first variable.
#
#     Parameters:
#     - x1: pd.Series, first continuous variable (original data).
#     - x2: pd.Series, second continuous variable (synthetic data).
#     - xsyn: pd.Series, all synthetic values for multiple syntheses.
#     - n: int, number of groups (bins) to create.
#     - style: str, method of grouping ('quantile' or 'equal').
#     - cont_na: list or pd.Series, values to be considered as missing/special.
#
#     Returns:
#     - list: [grouped_x1, grouped_x2], both as pd.Series with categorized values.
#     """
#
#     # Check if all inputs are numeric
#     if not (pd.api.types.is_numeric_dtype(x1) and
#             pd.api.types.is_numeric_dtype(x2) and
#             pd.api.types.is_numeric_dtype(xsyn)):
#         raise ValueError("x1, x2, and xsyn must be numeric.")
#
#     # If cont_na is not specified, initialize it as an empty list
#     if cont_na is None:
#         cont_na = []
#
#     # Select non-missing values (excluding continuous NA)
#     x1nm = x1[~x1.isin(cont_na) & x1.notna()]
#     x2nm = x2[~x2.isin(cont_na) & x2.notna()]
#     xsynnm = xsyn[~xsyn.isin(cont_na) & xsyn.notna()]
#
#     # Determine breaks based on the specified style
#     if style == "quantile":
#         # Unique breaks based on quantiles
#         my_breaks = np.unique(np.quantile(np.concatenate([x1nm, xsynnm]),
#                                           np.linspace(0, 1, n + 1)))
#     elif style == "equal":
#         # Unique breaks based on equal-width
#         my_breaks = np.unique(np.linspace(min(np.concatenate([x1nm, xsynnm])),
#                                           max(np.concatenate([x1nm, xsynnm])), n + 1))
#     else:
#         raise ValueError("Unknown style. Use 'quantile' or 'equal'.")
#
#     # Define group labels for the cuts
#     my_levels = list(pd.cut(x1nm, bins=my_breaks, right=False, include_lowest=True).cat.categories) + list(cont_na)
#
#     # Apply the grouping to non-missing data
#     x1_grouped = pd.cut(x1nm, bins=my_breaks, right=False, include_lowest=True).astype(str)
#     x2_grouped = pd.cut(x2nm, bins=my_breaks, right=False, include_lowest=True).astype(str)
#
#     # Map grouped values back to the original data including NAs
#     x1.loc[~x1.isin(cont_na) & x1.notna()] = x1_grouped
#     x2.loc[~x2.isin(cont_na) & x2.notna()] = x2_grouped
#
#     # Convert back to categorical with predefined levels
#     x1 = pd.Categorical(x1, categories=my_levels)
#     x2 = pd.Categorical(x2, categories=my_levels)
#
#     return [x1, x2]

import numpy as np
import pandas as pd

def group_num(x1, x2, xsyn, n=5, style='quantile', cont_na=None, **kwargs):
    # Ensure the inputs are numpy arrays
    x1 = np.array(x1)
    x2 = np.array(x2)
    xsyn = np.array(xsyn)

    if not (np.issubdtype(x1.dtype, np.number) and np.issubdtype(x2.dtype, np.number) and np.issubdtype(xsyn.dtype, np.number)):
        raise ValueError("x1, x2, and xsyn must be numeric.")

    # Handle missing values based on cont_na
    if cont_na is None:
        cont_na = {}  # Ensure cont_na is a dictionary if None

    x1nm = x1[~np.isnan(x1) & ~np.isin(x1, cont_na.get('x1', []))]
    x2nm = x2[~np.isnan(x2) & ~np.isin(x2, cont_na.get('x2', []))]
    xsynnm = xsyn[~np.isnan(xsyn) & ~np.isin(xsyn, cont_na.get('xsyn', []))]

    # Derive breaks based on style
    if style == 'quantile':
        combined = np.concatenate([x1nm, xsynnm])
        my_breaks = np.percentile(combined, np.linspace(0, 100, n + 1))
    elif style == 'equal':
        combined = np.concatenate([x1nm, xsynnm])
        my_breaks = np.linspace(np.min(combined), np.max(combined), n + 1)
    else:
        raise ValueError(f"Style '{style}' is not supported.")

    # Ensure unique breaks
    my_breaks = np.unique(my_breaks)

    # Define levels
    my_levels = list(pd.cut(x1nm, bins=my_breaks, labels=False, right=False, include_lowest=True).astype(str))
    my_levels = pd.Series(my_levels).unique().tolist()

    for name, values in cont_na.items():
        if values is not None:
            for val in values:
                my_levels.append(str(val))

    # Apply groupings to non-missing data
    x1_grouped = np.where(~np.isnan(x1) & ~np.isin(x1, cont_na.get('x1', [])),
                          pd.cut(x1nm, bins=my_breaks, labels=False, right=False, include_lowest=True).astype(str),
                          str(cont_na.get('x1', [np.nan])[0]))
    x2_grouped = np.where(~np.isnan(x2) & ~np.isin(x2, cont_na.get('x2', [])),
                          pd.cut(x2nm, bins=my_breaks, labels=False, right=False, include_lowest=True).astype(str),
                          str(cont_na.get('x2', [np.nan])[0]))

    # Convert to categorical data
    x1 = pd.Categorical(x1_grouped, categories=my_levels)
    x2 = pd.Categorical(x2_grouped, categories=my_levels)

    return x1, x2

