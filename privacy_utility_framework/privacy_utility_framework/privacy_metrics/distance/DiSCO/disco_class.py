import pandas as pd
from sklearn.neighbors import NearestNeighbors

from privacy_utility_framework.privacy_utility_framework.privacy_metrics import PrivacyMetricCalculator


class DiscoCalculator(PrivacyMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame):
        super().__init__(original, synthetic)

    def evaluate(self):
        pass
        # data is a dataframe
        # dd <- data
        # Nd <- dim(dd)[1]

        # tab_ktd <- table(dd$target,dd$keys)   ## two way target and keys table orig

        #  if (!(all(Ks %in% Kd))) {  ## extra synthetic keys not in original
        # extraKs <- Ks[!(Ks %in% Kd) ]
        # extra_tab <- matrix(0,dim(tab_ktd)[1],length(extraKs))
        # dimnames(extra_tab) <- list(dimnames(tab_ktd)[[1]],extraKs)
        # tab_ktd <- cbind(tab_ktd,extra_tab)
        # tab_ktd <- tab_ktd[,order(dimnames(tab_ktd)[[2]]), drop = FALSE]

        # tab_iS <- tab_ktd
        # tab_DiSCO <- tab_iS
        # DiSCO <- sum(tab_DiSCO)/Nd*100
        # Perform the evaluation and return the result
        import pandas as pd

    # def calculate_disco(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, key_columns: list) -> float:
    """
    Calculate the DiSCO (Disclosure Risk) metric.

    Parameters:
    original_df (pd.DataFrame): The original dataset.
    synthetic_df (pd.DataFrame): The synthetic dataset.
    key_columns (list): List of column names that are considered as key attributes.

    Returns:
    float: The DiSCO value.
    """

    # def _compute_distance_matrix(df: pd.DataFrame, key_columns: list) -> np.ndarray:
    #     """Compute distance matrix for given DataFrame based on key columns."""
    #     X = df[key_columns].values
    #     nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X)
    #     distances, _ = nn.kneighbors(X)
    #     return distances
    #
    # # Compute distance matrices
    # dist_matrix_original = _compute_distance_matrix(original_df, key_columns)
    # dist_matrix_synthetic = _compute_distance_matrix(synthetic_df, key_columns)
    #
    # # Compute DiSCO
    # min_dist_original = np.min(dist_matrix_original, axis=1)
    # min_dist_synthetic = np.min(dist_matrix_synthetic, axis=1)
    #
    # disco_value = np.mean(min_dist_synthetic > min_dist_original) * 100
    #
    # return disco_value
    import pandas as pd

def disco_test2(original_data, synthetic_data, target, keys):

    # Assuming object, data, target, and keys are provided
    syndata = synthetic_data

    # Rename target variable to 'target'
    dd = original_data.copy()
    dd['target'] = dd[target]

    # Restrict dd to target and keys
    dd = dd[['target'] + keys]

    # Restrict syndata to target and keys
    syndata['target'] = syndata[target]
    ss = syndata[['target'] + keys]

    # Combine keys into a single column
    if len(keys) > 1:
        ss['keys'] = ss[keys].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)
        dd['keys'] = dd[keys].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)
    else:
        ss['keys'] = ss[keys[0]]
        dd['keys'] = dd[keys[0]]

    # Create contingency tables (equivalent to table() in R)
    tab_kts = pd.crosstab(ss['target'], ss['keys'])
    tab_ktd = pd.crosstab(dd['target'], dd['keys'])

    # Calculate proportions
    tab_ktd_p = tab_ktd.div(tab_ktd.sum(axis=0), axis=1).fillna(0)
    tab_kts_p = tab_kts.div(tab_kts.sum(axis=0), axis=1).fillna(0)

    # Create tab_iS and set keys_syn = 0
    keys_syn = tab_kts.sum(axis=0)
    tab_iS = tab_ktd.copy()
    print(f'Shape: {tab_iS.shape}')

    # TODO: fix and adjust for no target given
    print(f'Len: {keys_syn}')
    tab_iS.loc[:, keys_syn == 0] = 0

    # Calculate DiSCO
    tab_DiSCO = tab_iS.copy()
    tab_DiSCO.loc[~tab_kts_p.eq(1).all(axis=1)] = 0
    # Final values for tab_ks, tab_kd, repU, and DiSCO
    tab_ks = tab_kts.sum(axis=0)
    tab_kd = tab_ktd.sum(axis=0)

    tab_ksd1 = tab_kd[(tab_ks == 1) & (tab_kd == 1)]  # repU

    Nd = len(dd)
    repU = tab_ksd1.sum() / Nd * 100
    DiSCO = tab_DiSCO.sum().sum() / Nd * 100

    # Print the results
    print(f"repU: {repU}")
    print(f"DiSCO: {DiSCO}")

def disco_test1(original_data, synthetic_data, target_variable, key_variables):
    all_disco_measures = []

        # Rename target variable in `data` to `target`
    dd = original_data.copy()  # Copy the DataFrame to avoid changing the original
    dd['target'] = dd[target]  # Create a new 'target' column

    # Subset the data to include only the 'target' and 'keys' columns
    dd = dd[['target'] + key_variables]

    ss = synthetic_data.copy()
    ss['target'] = ss[target_variable]
    ss = ss[['target'] + key_variables]
# Assuming `ss` and `dd` are pandas DataFrames and `keys` is a list of column names
    if len(keys) > 1:
        # Create a composite key by concatenating values of the key columns, separated by ' | '
        ss['keys'] = ss[keys].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)
        dd['keys'] = dd[keys].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)
    else:
        # If only one key column, just assign that column as 'keys'
        ss['keys'] = ss[keys[0]]
        dd['keys'] = dd[keys[0]]

    NKd = len(pd.crosstab(dd['keys']))
    #NKs <- length(table(ss$keys))
    # Calculate DiSCO for each synthetic dataset
    # Create contingency tables
    tab_ktd = pd.crosstab(dd['target'], dd['keys'])
    tab_kts = pd.crosstab(ss['target'], ss['keys'])

    tab_ktd_p = tab_ktd.div(tab_ktd.sum(axis=1), axis=0)
    tab_kts_p = tab_kts.div(tab_kts.sum(axis=1), axis=0)

    # Calculate intermediate values
    did = tab_ktd.copy()
    did[tab_ktd_p != 1] = 0
    dis = tab_kts.copy()
    dis[tab_kts_p != 1] = 0
    keys_syn = tab_kts.sum(axis=0)
    tab_iS = tab_ktd.copy()
    tab_iS.loc[:, keys_syn == 0] = 0
    tab_DiS = tab_ktd.copy()
    anydis = tab_kts_p.apply(lambda x: any(x == 1), axis=1)
    tab_DiS.loc[:, ~anydis] = 0
    tab_DiSCO = tab_iS.copy()
    tab_DiSCO.loc[tab_kts_p != 1] = 0
    tab_DiSDiO = tab_DiSCO.copy()
    tab_DiSDiO.loc[tab_ktd_p != 1] = 0

    # Calculate DiSCO
    disco = sum(tab_DiSCO) / tab_ktd.shape[0] * 100

    print(f"DiSCO: {disco:.2f}")
    # # Calculate proportions and margins
    # dd_contingency_table_proportions = dd_contingency_table.div(dd_contingency_table.sum(axis=1), axis=0)
    # dd_contingency_table_proportions.fillna(0, inplace=True)
    # ss_contingency_table_proportions = ss_contingency_table.div(ss_contingency_table.sum(axis=1), axis=0)
    # ss_contingency_table_proportions.fillna(0, inplace=True)
    #
    # dd_marginal_totals = dd_contingency_table.sum(axis=0)
    # ss_marginal_totals = ss_contingency_table.sum(axis=0)
    # total_original = dd_contingency_table.sum().sum()
    # total_synthetic = ss_contingency_table.sum().sum()
    #
    # # Calculate disclosure measures
    # did = dd_contingency_table.copy()
    # did.loc[(dd_contingency_table_proportions != 1).all(axis=1), :] = 0
    # dis = ss_contingency_table.copy()
    # dis.loc[(ss_contingency_table_proportions != 1).all(axis=1), :] = 0
    # keys_syn = ss_contingency_table.sum(axis=0)
    # tab_iS = dd_contingency_table.copy()
    # tab_iS.loc[keys_syn == 0, :] = 0
    # tab_DiS = dd_contingency_table.copy()
    # any_dis = ss_contingency_table_proportions.apply(pd.Series.any, axis=1)
    # tab_DiS.loc[~any_dis, :] = 0
    # tab_DiSCO = tab_iS.copy()
    # tab_DiSCO.loc[(ss_contingency_table_proportions != 1).all(axis=1), :] = 0
    # tab_DiSDiO = tab_DiSCO.copy()
    # tab_DiSDiO.loc[(dd_contingency_table_proportions != 1).all(axis=1), :] = 0
    #
    # # Calculate DiSCO and other measures
    # Dorig = (did.sum().sum() / total_original) * 100
    # Dsyn = (dis.sum().sum() / total_synthetic) * 100
    # iS = (tab_iS.sum().sum() / total_original) * 100
    # DiS = (tab_DiS.sum().sum() / total_original) * 100
    # DiSCO = (tab_DiSCO.sum().sum() / total_original) * 100
    # DiSDiO = (tab_DiSDiO.sum().sum() / total_original) * 100
    #
    # # Store measures in a dictionary
    # attrib = {
    #     "Dorig": Dorig,
    #     "Dsyn": Dsyn,
    #     "iS": iS,
    #     "DiS": DiS,
    #     "DiSCO": DiSCO,
    #     "DiSDiO": DiSDiO,
    #     "max_denom": max(tab_DiSCO),
    #     "mean_denom": tab_DiSCO[tab_DiSCO > 0].mean()
    # }
    #
    # # Append measures to the list
    # all_disco_measures.append(attrib)

    return all_disco_measures


def calculate_disco(original_data, synthetic_data, target_var, key_vars):
    """
    Calculates DiSCO for a given original and synthetic dataset.

    Args:
        original_data_file (pd): original data as pandas dataframe.
        synthetic_data_file (pd): synthetic data as pandas dataframe.
        target_var (str): Name of the target variable.
        key_vars (list): List of key variable names.

    Returns:
        tuple: A tuple containing the DiSCO value, a dictionary of intermediate results, and a dictionary of exclusion counts.
        :param original_data:
        :param synthetic_data:
        :param key_vars:
        :param target_var:
    """

    # Handle missing values and convert to appropriate data types (if necessary)
    # original_data.fillna(value="Missing", inplace=True)
    # synthetic_data.fillna(value="Missing", inplace=True)

    target_columns = original_data[target_var]
    keys_columns = original_data[key_vars]
    print(target_columns)
    print(keys_columns)
    # Create contingency tables
    tab_ktd = pd.crosstab(target_columns, keys_columns)
    tab_kts = pd.crosstab(synthetic_data[target_var], synthetic_data[key_vars])

    # Calculate proportions
    tab_ktd_p = tab_ktd.div(tab_ktd.sum(axis=1), axis=0)
    tab_kts_p = tab_kts.div(tab_kts.sum(axis=1), axis=0)

    # Calculate intermediate values
    did = tab_ktd.copy()
    did[tab_ktd_p != 1] = 0
    dis = tab_kts.copy()
    dis[tab_kts_p != 1] = 0
    keys_syn = tab_kts.sum(axis=0)
    tab_iS = tab_ktd.copy()
    tab_iS.loc[:, keys_syn == 0] = 0
    tab_DiS = tab_ktd.copy()
    anydis = tab_kts_p.apply(lambda x: any(x == 1), axis=1)
    tab_DiS.loc[:, ~anydis] = 0
    tab_DiSCO = tab_iS.copy()
    tab_DiSCO.loc[tab_kts_p != 1] = 0
    tab_DiSDiO = tab_DiSCO.copy()
    tab_DiSDiO.loc[tab_ktd_p != 1] = 0

    # Calculate DiSCO
    disco = sum(tab_DiSCO) / tab_ktd.shape[0] * 100

    # Calculate other metrics (if needed)
    # ...

    # Calculate exclusion counts (if needed)
    # ...

    # Return results
    results = {
        "disco": disco,
        # ... other intermediate results
    }
    return results


# NOTE: Still testing
real_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/transformed_SD2011_selected_columns.csv')
synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/transformed_syn_SD2011_selected_columns.csv')

keys = ["sex", "age", "region", "placesize"]
target = ["depress"]
d = disco_test2(real_data, synthetic_data, target, keys)
print(d)
#disco_value = calculate_disco(real_data, synthetic_data, target, keys)
#print(f"DiSCO Value: {disco_value:.2f}%")
