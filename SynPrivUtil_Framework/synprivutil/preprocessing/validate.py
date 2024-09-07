
def validate_datasets(original_df, synthetic_df):
    # Check column names and count
    assert set(original_df.columns) == set(synthetic_df.columns), "Column names do not match."
    assert len(original_df.columns) == len(synthetic_df.columns), "Number of columns does not match."

    #Check data types
    for column in original_df.columns:
        assert original_df[column].dtype == synthetic_df[column].dtype, f"Data type mismatch in column {column}."

    #Check for missing values
    assert not original_df.isnull().any().any(), "Original dataset contains missing values."
    assert not synthetic_df.isnull().any().any(), "Synthetic dataset contains missing values."

    for column in original_df.select_dtypes(include=['category', 'object']).columns:
        assert set(synthetic_df[column].unique()).issubset(set(original_df[column].unique())), f"Unexpected values in {column}."

    return True  #If all checks pass
