import pandas as pd
from anonymeter.evaluators import InferenceEvaluator

from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator


class DiscoCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        super().__init__(original_data, synthetic_data)

    def evaluate(self):
        pass
        # Create an instance of InferenceCalculator with the provided columns
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
import numpy as np

def calculate_disco(original_df, synthetic_df, key_columns):
    # Ensure key_columns exist in both datasets
    assert all(col in original_df.columns for col in key_columns)
    assert all(col in synthetic_df.columns for col in key_columns)

    # Create a cross-tabulation (joint distribution) on key columns
    original_ctab = pd.crosstab(index=[original_df[key] for key in key_columns], columns='count')
    synthetic_ctab = pd.crosstab(index=[synthetic_df[key] for key in key_columns], columns='count')

    # Align the synthetic crosstab to the original (add missing keys)
    synthetic_ctab = synthetic_ctab.reindex(original_ctab.index, fill_value=0)

    # Calculate DiSCO (sum of correctly matched counts normalized by the original's total)
    disco = (synthetic_ctab * original_ctab).sum().sum() / original_ctab.sum().sum() * 100

    return disco


real_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/SD2011_selected_columns.csv')
synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/syn_SD2011_selected_columns.csv')

key_columns = ["sex", "age", "region", "placesize"]
disco_value = calculate_disco(real_data, synthetic_data, key_columns)
print(f"DiSCO Value: {disco_value:.2f}%")

