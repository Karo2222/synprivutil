import pandas as pd
import numpy as np


# def synorig_compare(syn, orig, print_flag=True):
#     needs_fix = False
#     unchanged = True
#
#     # Convert to DataFrame if not already
#     if not isinstance(syn, pd.DataFrame):
#         syn = pd.DataFrame(syn)
#         unchanged = False
#     if not isinstance(orig, pd.DataFrame):
#         orig = pd.DataFrame(orig)
#         unchanged = False
#
#     # Check for variables in `syn` but not in `orig`
#     syn_vars = set(syn.columns)
#     orig_vars = set(orig.columns)
#     extra_vars_in_syn = syn_vars - orig_vars
#     if extra_vars_in_syn:
#         print(f"Variables {', '.join(extra_vars_in_syn)} in synthetic but not in original")
#         syn = syn.loc[:, orig_vars]
#         print(f"{', '.join(extra_vars_in_syn)} dropped from syn\n\n")
#
#     # Reduce `syn` and `orig` to common variables in the same order
#     common_vars = list(orig_vars & syn_vars)
#     len_common = len(common_vars)
#     if print_flag:
#         print(f"{len_common} variables in common out of {len(syn.columns)} in syn out of {len(orig.columns)} in orig")
#
#     orig = orig[common_vars]
#     syn = syn[common_vars]
#
#     # Change common variables that are numeric in `syn` and factors in `orig` to factors in `syn`
#     nch_syn = 0
#     nch_orig = 0
#     for i in range(len_common):
#         if pd.api.types.is_numeric_dtype(syn.iloc[:, i]) and pd.api.types.is_categorical_dtype(orig.iloc[:, i]):
#             syn.iloc[:, i] = pd.Categorical(syn.iloc[:, i])
#             nch_syn += 1
#             unchanged = False
#         if pd.api.types.is_categorical_dtype(syn.iloc[:, i]) and pd.api.types.is_numeric_dtype(orig.iloc[:, i]):
#             orig.iloc[:, i] = pd.Categorical(orig.iloc[:, i])
#             nch_orig += 1
#             unchanged = False
#
#     if not unchanged:
#         print(f"\nVariables changed from numeric to factor {nch_syn} in syn {nch_orig} in original\n\n")
#
#     # Change common character variables to factors
#     nch_syn = 0
#     nch_orig = 0
#     unchanged2 = True
#     for i in range(len_common):
#         if pd.api.types.is_string_dtype(syn.iloc[:, i]):
#             syn.iloc[:, i] = pd.Categorical(syn.iloc[:, i])
#             nch_syn += 1
#             unchanged2 = False
#             unchanged = False
#         if pd.api.types.is_string_dtype(orig.iloc[:, i]):
#             orig.iloc[:, i] = pd.Categorical(orig.iloc[:, i])
#             nch_orig += 1
#             unchanged2 = False
#             unchanged = False
#
#     if not unchanged2:
#         print(f"\nVariables changed from character to factor x {nch_syn} in syn and {nch_orig} in orig\n\n")
#
#     # Check data types match in common variables
#     for i in range(len_common):
#         if pd.api.types.is_integer_dtype(syn.iloc[:, i]) and pd.api.types.is_numeric_dtype(orig.iloc[:, i]) and not pd.api.types.is_integer_dtype(orig.iloc[:, i]):
#             syn.iloc[:, i] = syn.iloc[:, i].astype(float)
#             print(f"{syn.columns[i]} changed from integer to numeric in synthetic to match original")
#             unchanged = False
#         elif pd.api.types.is_integer_dtype(orig.iloc[:, i]) and pd.api.types.is_numeric_dtype(syn.iloc[:, i]) and not pd.api.types.is_integer_dtype(syn.iloc[:, i]):
#             orig.iloc[:, i] = orig.iloc[:, i].astype(float)
#             print(f"{orig.columns[i]} changed from integer to numeric in original to match synthetic")
#             unchanged = False
#         elif syn.dtypes[i] != orig.dtypes[i]:
#             print(f"\nDifferent types for {syn.columns[i]} in syn: {syn.dtypes[i]} in orig: {orig.dtypes[i]}\n")
#             needs_fix = True
#
#     # Compare missingness and levels for factors
#     for i in range(len_common):
#         if not orig.iloc[:, i].isna().any() and syn.iloc[:, i].isna().any():
#             print(f"\n\nMissing data for common variable {syn.columns[i]} in syn but not in orig\nThis looks wrong check carefully\n")
#             orig.iloc[:, i] = orig.iloc[:, i].astype('category').cat.add_categories('Missing')
#             print(f"NA added to factor {orig.columns[i]} in orig\n\n\n")
#             unchanged = False
#         if pd.api.types.is_categorical_dtype(syn.iloc[:, i]) and pd.api.types.is_categorical_dtype(orig.iloc[:, i]):
#             if orig.iloc[:, i].isna().any() and not syn.iloc[:, i].isna().any():
#                 syn.iloc[:, i] = syn.iloc[:, i].astype('category').cat.add_categories('Missing')
#                 print(f"NA added to factor {syn.columns[i]} in syn\n")
#                 unchanged = False
#             lev1 = syn.iloc[:, i].cat.categories
#             lev2 = orig.iloc[:, i].cat.categories
#
#             if len(lev1) != len(lev2) or not all(lev1 == lev2):
#                 print(f"\nFactor levels don't match for {syn.columns[i]} levels combined\n"
#                       f"syn levels {lev1}\n"
#                       f"orig levels {lev2}\n")
#
#                 all_levels = lev1.union(lev2)
#                 syn.iloc[:, i] = pd.Categorical(syn.iloc[:, i], categories=all_levels, ordered=False)
#                 orig.iloc[:, i] = pd.Categorical(orig.iloc[:, i], categories=all_levels, ordered=False)
#                 unchanged = False
#
#     if needs_fix:
#         print("\n***********************************************************************************\n"
#               "STOP: you may need to change the original or synthetic data to make them match:\n")
#
#     if not unchanged:
#         print("\n*****************************************************************\n"
#               "Differences detected and corrections attempted check output above.\n")
#     else:
#         print("Synthetic and original data checked with synorig.compare,\n looks like no adjustment needed\n\n")
#
#     return {'syn': syn, 'orig': orig, 'needs_fix': needs_fix, 'unchanged': unchanged}


def synorig_compare2(syn, orig, print_flag=True):
    needsfix = False
    unchanged = True

    # Convert any tibbles or matrices
    if not isinstance(syn, pd.DataFrame):
        syn = pd.DataFrame(syn)
        unchanged = False
    if not isinstance(orig, pd.DataFrame):
        orig = pd.DataFrame(orig)
        unchanged = False

    # Check for variables in synthetic but not in original
    if any(col not in orig.columns for col in syn.columns):
        out = [col for col in syn.columns if col not in orig.columns]
        print(f"Variables {out} in synthetic but not in original")
        syn = syn.loc[:, syn.columns.isin(orig.columns)]
        print(f"{out} dropped from syn\n")

    # Reduce syn and orig to common vars in same order
    common = orig.columns[orig.columns.isin(syn.columns)]
    len_common = len(common)
    if print_flag:
        print(f"{len_common} variables in common out of {len(syn.columns)} in syn out of {len(orig.columns)} in orig")

    #Reorder to match up
    orig = orig.loc[:, orig.columns.isin(common)]
    syn = syn.loc[:, syn.columns.isin(common)]
    syn = syn.loc[:, orig.columns]


    # Change common variables that are numeric in syn and factors in orig to factors in syn
    nch_syn = 0
    nch_orig = 0
    for i in range(len_common):
        print(f'cat 1 {orig.iloc[:, i].dtype}')
        print(orig.iloc[:, i])
        if pd.api.types.is_numeric_dtype(syn.iloc[:, i]) and isinstance(orig.iloc[:, i].dtype, pd.CategoricalDtype):
            syn.iloc[:, i] = syn.iloc[:, i].astype('category')
            nch_syn += 1
            unchanged = False
        if isinstance(syn.iloc[:, i].dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(orig.iloc[:, i]):
            orig.iloc[:, i] = orig.iloc[:, i].astype('category')
            nch_orig += 1
            unchanged = False
    if not unchanged:
        print(f"\nVariables changed from numeric to factor {nch_syn} in syn {nch_orig} in original\n")

    # Change common character variables to factors
    nch_syn = 0
    nch_orig = 0
    unchanged2 = True
    for i in range(len_common):
        if pd.api.types.is_string_dtype(syn.iloc[:, i]):
            # syn.loc[:, i] = syn.iloc[:, i].astype('category')
            syn.iloc[:, i] = pd.Categorical(syn.iloc[:, i])
            nch_syn += 1
            unchanged2 = False
            unchanged = False
        if pd.api.types.is_string_dtype(orig.iloc[:, i]):
            #orig.loc[:, i] = orig.iloc[:, i].astype('category')
            orig.iloc[:, i] = pd.Categorical(orig.iloc[:, i])
            nch_orig += 1
            unchanged2 = False
            unchanged = False
    if not unchanged2:
        print(f"\nVariables changed from character to factor {nch_syn} in syn and {nch_orig} in orig\n")

    # Check data types match in common variables
    for i in range(len_common):
        if pd.api.types.is_integer_dtype(syn.iloc[:, i]) and pd.api.types.is_float_dtype(orig.iloc[:, i]):
            syn.iloc[:, i] = pd.to_numeric(syn.iloc[:, i])
            print(f"{syn.columns[i]} changed from integer to numeric in synthetic to match original")
            unchanged = False
        elif pd.api.types.is_integer_dtype(orig.iloc[:, i]) and pd.api.types.is_float_dtype(syn.iloc[:, i]):
            orig.iloc[:, i] = pd.to_numeric(orig.iloc[:, i])
            print(f"{orig.columns[i]} changed from integer to numeric in original to match synthetic")
            unchanged = False
        elif syn.iloc[:, i].dtype != orig.iloc[:, i].dtype:
            print(f'syn: {syn.iloc[:, i].dtype}')
            print(f'orig: {orig.iloc[:, i].dtype}')
            print(f"\nDifferent classes for {syn.columns[i]} in syn: {syn.iloc[:, i].dtype} in orig: {orig.iloc[:, i].dtype}\n")
            needsfix = True

    # Compare missingness and levels for factors
    for i in range(len_common):
        if not orig.iloc[:, i].isna().any() and syn.iloc[:, i].isna().any():
            print(f"\n\nMissing data for common variable {syn.columns[i]} in syn but not in orig\nThis looks wrong check carefully\n")
            print("PD CATEGORICAL 5")
            #orig.iloc[:, i] = orig.iloc[:, i].astype('category').cat.add_categories([pd.NA])
            orig.iloc[:, i] = orig.iloc[:, i].astype('category')
            orig.iloc[:, i] = orig.iloc[:, i].fillna(value=pd.NA)

            print(f"NA added to factor {orig.columns[i]} in orig\n\n\n")
            unchanged = False
        if isinstance(syn.iloc[:, i].dtype, pd.CategoricalDtype) and isinstance(orig.iloc[:, i].dtype, pd.CategoricalDtype):
            if orig.iloc[:, i].isna().any() and not syn.iloc[:, i].isna().any():
                print("PD CATEGORICAL 6")
                syn.iloc[:, i] = syn.iloc[:, i].astype('category').cat.add_categories([pd.NA])
                print(f"NA added to factor {syn.columns[i]} in syn")
                unchanged = False
            lev1 = syn.iloc[:, i].cat.categories
            lev2 = orig.iloc[:, i].cat.categories
            if len(lev1) != len(lev2) or not np.array_equal(lev1, lev2):
                print(f"\nDifferent levels for {syn.columns[i]} in syn: {lev1} in orig: {lev2}\n")
                all_levels = lev1.union(lev2)
                syn.iloc[:, i] = pd.Categorical(syn.iloc[:, i], categories=all_levels, ordered=False)
                orig.iloc[:, i] = pd.Categorical(orig.iloc[:, i], categories=all_levels, ordered=False)
                unchanged = False
    if needsfix:
        print("\n***********************************************************************************\n"
              "STOP: you may need to change the original or synthetic data to make them match:\n")

    if not unchanged:
        print("\n*****************************************************************\n"
              "Differences detected and corrections attempted check output above.\n")
    else:
        print("Synthetic and original data checked with synorig.compare,\n looks like no adjustment needed\n\n")

    return {'syn': syn, 'orig': orig, 'needs_fix': needsfix, 'unchanged': unchanged}


def test_synorig_compare2():
    # Test 1: Convert any tibbles or matrices
    syn = np.array([[1, 2], [3, 4]])
    orig = np.array([[1, 2], [3, 4]])
    print("Test 1: Convert any tibbles or matrices")
    synorig_compare2(syn, orig)

    # Test 2: Check for variables in synthetic but not in original
    syn = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    orig = pd.DataFrame({'A': [1, 2]})
    print("\nTest 2: Check for variables in synthetic but not in original")
    synorig_compare2(syn, orig)

    # Test 3: Change common variables that are numeric in syn and factors in orig to factors in syn
    syn = pd.DataFrame({'A': [1, 2, 3]})
    orig = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'c'])})
    print("\nTest 3: Change common variables that are numeric in syn and factors in orig to factors in syn")
    synorig_compare2(syn, orig)

    # Test 4: Change common variables that are factors in syn and numeric in orig to factors in orig
    syn = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'c'])})
    orig = pd.DataFrame({'A': [1, 2, 3]})
    print("\nTest 4: Change common variables that are factors in syn and numeric in orig to factors in orig")
    synorig_compare2(syn, orig)

    # Test 5: Change common character variables to factors in syn
    syn = pd.DataFrame({'A': ['a', 'b', 'c']})
    orig = pd.DataFrame({'A': ['a', 'b', 'c']})
    print("\nTest 5: Change common character variables to factors in syn")
    synorig_compare2(syn, orig)

    # Test 6: Change common character variables to factors in orig
    syn = pd.DataFrame({'A': ['a', 'b', 'c']})
    orig = pd.DataFrame({'A': ['a', 'b', 'c']})
    print("\nTest 6: Change common character variables to factors in orig")
    synorig_compare2(syn, orig)

    # Test 7: Check data types match in common variables (integer to numeric in syn)
    syn = pd.DataFrame({'A': [1, 2, 3]})
    orig = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
    print("\nTest 7: Check data types match in common variables (integer to numeric in syn)")
    synorig_compare2(syn, orig)

    # Test 8: Check data types match in common variables (integer to numeric in orig)
    syn = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
    orig = pd.DataFrame({'A': [1, 2, 3]})
    print("\nTest 8: Check data types match in common variables (integer to numeric in orig)")
    synorig_compare2(syn, orig)

    # Test 9: Compare missingness and levels for factors (missing data in syn but not in orig)
    syn = pd.DataFrame({'A': [1, np.nan, 3]})
    orig = pd.DataFrame({'A': [1, 2, 3]})
    print("\nTest 9: Compare missingness and levels for factors (missing data in syn but not in orig)")
    synorig_compare2(syn, orig)

    # Test 10: Compare missingness and levels for factors (missing data in orig but not in syn)
    syn = pd.DataFrame({'A': [1, 2, 3]})
    orig = pd.DataFrame({'A': [1, np.nan, 3]})
    print("\nTest 10: Compare missingness and levels for factors (missing data in orig but not in syn)")
    synorig_compare2(syn, orig)

    # Test 11: Compare levels for factors (different levels in syn and orig)
    syn = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'c'])})
    orig = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'd'])})
    print("\nTest 11: Compare levels for factors (different levels in syn and orig)")
    synorig_compare2(syn, orig)

# Call the test function
#test_synorig_compare2()

# Example usage
# syn = pd.DataFrame(...)
# orig = pd.DataFrame(...)
# synorig_compare(syn, orig)
