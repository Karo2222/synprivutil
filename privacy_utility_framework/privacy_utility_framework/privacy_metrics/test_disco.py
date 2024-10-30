# does this look ok:
#
# # ... other function code ...
#
# ###--------------------- make tables ---------------------------
#
# NKd <- length(table(dd$keys))
# NKs <- length(table(ss$keys))
#
# tab_kts <- table(ss$target, ss$keys)
# tab_ktd <- table(dd$target, dd$keys)  ## two way target and keys table orig
#
# # ... other code ...
#
# ###------------------------- calculate proportions and margins ---------
#
# tab_ktd_p <- sweep(tab_ktd,2,apply(tab_ktd,2,sum),"/")
# tab_ktd_p[is.na(tab_ktd_p)] <- 0
# tab_kts_p <- sweep(tab_kts,2,apply(tab_kts,2,sum),"/")
# tab_kts_p[is.na(tab_kts_p)] <- 0
#
# tab_kd <- apply(tab_ktd,2,sum)
# tab_td<- apply(tab_ktd,1,sum)
# tab_ks <- apply(tab_kts,2,sum)
# tab_ts <- apply(tab_kts,1,sum)
# #
# # ... other code ...
#
# ###------------------------get tables for  calculating attribute disclosure measures-----------------------------------------------------
#
# did <- tab_ktd ; did[tab_ktd_p != 1] <- 0
# dis <- tab_kts ; dis[tab_kts_p != 1] <- 0
# keys_syn <- apply(tab_kts,2,sum)
# tab_iS <- tab_ktd
# tab_iS[,keys_syn ==0 ] <- 0
# tab_DiS <- tab_ktd
# anydis <- apply(tab_kts_p,2,function(x) any(x ==1))
# tab_DiS[,!anydis] <- 0
# tab_DiSCO <- tab_iS
# tab_DiSCO[tab_kts_p != 1] <- 0
# tab_DiSDiO <- tab_DiSCO
# tab_DiSDiO[tab_ktd_p != 1] <- 0
#
# # ... other code ...
#
# ###----------------------------- attrib dis measures-------------------------
# Dorig <- sum(did)/Nd*100
# Dsyn <- sum(dis)/Ns*100
# iS <- sum(tab_iS)/Nd*100
# DiS <- sum(tab_DiS)/Nd*100
# DiSCO <- sum(tab_DiSCO)/Nd*100  # This line calculates DiSCO
# DiSDiO <- sum(tab_DiSDiO)/Nd*100
#
# attrib[jj,] <- c( Dorig,Dsyn,iS,DiS,DiSCO,DiSDiO,max(tab_DiSCO),mean(tab_DiSCO[tab_DiSCO>0]))
#
# # ... other function code ...
#
#
#
# import pandas as pd
#
# def calculate_disco(original_data, synthetic_data, target_variable, key_variables):
#     """
#     Calculates DiSCO (Distance Components disclosure) for a set of synthetic data.
#
#     Args:
#       original_data: Pandas DataFrame containing the original data.
#       synthetic_data: List of Pandas DataFrames, each representing a synthetic dataset.
#       target_variable: String, the name of the target variable in the data.
#       key_variables: List of strings, the names of the key variables used for identification.
#
#     Returns:
#       A list of dictionaries, each containing DiSCO and other attribute disclosure measures
#       for a single synthetic dataset.
#     """
#
#     all_disco_measures = []
#     for ss in synthetic_data:
#         # Create contingency tables
#         dd_contingency_table = pd.crosstab(original_data[target_variable], original_data[key_variables])
#         ss_contingency_table = pd.crosstab(ss[target_variable], ss[key_variables])
#
#         # Calculate proportions and margins
#         dd_contingency_table_proportions = dd_contingency_table.div(dd_contingency_table.sum(axis=1), axis=0)
#         dd_contingency_table_proportions.fillna(0, inplace=True)
#         ss_contingency_table_proportions = ss_contingency_table.div(ss_contingency_table.sum(axis=1), axis=0)
#         ss_contingency_table_proportions.fillna(0, inplace=True)
#
#         dd_marginal_totals = dd_contingency_table.sum(axis=0)
#         ss_marginal_totals = ss_contingency_table.sum(axis=0)
#         total_original = dd_contingency_table.sum().sum()
#         total_synthetic = ss_contingency_table.sum().sum()
#
#         # Calculate disclosure measures
#         did = dd_contingency_table.copy()
#         did.loc[(dd_contingency_table_proportions != 1).all(axis=1), :] = 0
#         dis = ss_contingency_table.copy()
#         dis.loc[(ss_contingency_table_proportions != 1).all(axis=1), :] = 0
#         keys_syn = ss_contingency_table.sum(axis=0)
#         tab_iS = dd_contingency_table.copy()
#         tab_iS.loc[keys_syn == 0, :] = 0
#         tab_DiS = dd_contingency_table.copy()
#         any_dis = ss_contingency_table_proportions.apply(pd.Series.any, axis=1)
#         tab_DiS.loc[~any_dis, :] = 0
#         tab_DiSCO = tab_iS.copy()
#         tab_DiSCO.loc[(ss_contingency_table_proportions != 1).all(axis=1), :] = 0
#         tab_DiSDiO = tab_DiSCO.copy()
#         tab_DiSDiO.loc[(dd_contingency_table_proportions != 1).all(axis=1), :] = 0
#
#         # Calculate DiSCO and other measures
#         Dorig = (did.sum().sum() / total_original) * 100
#         Dsyn = (dis.sum().sum() / total_synthetic) * 100
#         iS = (tab_iS.sum().sum() / total_original) * 100
#         DiS = (tab_DiS.sum().sum() / total_original) * 100
#         DiSCO = (tab_DiSCO.sum().sum() / total_original) * 100
#         DiSDiO = (tab_DiSDiO.sum().sum() / total_original) * 100
#
#         # Store measures in a dictionary
#         attrib = {
#             "Dorig": Dorig,
#             "Dsyn": Dsyn,
#             "iS": iS,
#             "DiS": DiS,
#             "DiSCO": DiSCO,
#             "DiSDiO": DiSDiO,