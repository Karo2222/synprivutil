import pandas as pd
from sklearn.neighbors import NearestNeighbors

from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator


class DiscoCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        super().__init__(original_data, synthetic_data)

    def evaluate(self):
        pass
        #data is a dataframe
        #dd <- data
        #Nd <- dim(dd)[1]


        # tab_ktd <- table(dd$target,dd$keys)   ## two way target and keys table orig

        #  if (!(all(Ks %in% Kd))) {  ## extra synthetic keys not in original
        # extraKs <- Ks[!(Ks %in% Kd) ]
        # extra_tab <- matrix(0,dim(tab_ktd)[1],length(extraKs))
        # dimnames(extra_tab) <- list(dimnames(tab_ktd)[[1]],extraKs)
        # tab_ktd <- cbind(tab_ktd,extra_tab)
        # tab_ktd <- tab_ktd[,order(dimnames(tab_ktd)[[2]]), drop = FALSE]

        #tab_iS <- tab_ktd
        #tab_DiSCO <- tab_iS
        #DiSCO <- sum(tab_DiSCO)/Nd*100
        # Perform the evaluation and return the result
        import pandas as pd

def calculate_disco(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, key_columns: list) -> float:
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

# NOTE: Still testing
real_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/SD2011_selected_columns.csv')
synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/syn_SD2011_selected_columns.csv')

key_columns = ["sex", "age", "region", "placesize"]
disco_value = calculate_disco(real_data, synthetic_data, key_columns)
print(f"DiSCO Value: {disco_value:.2f}%")

