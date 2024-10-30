# # import numpy as np
# #
# #
# # def tab_exclude(xx, col, Nexcludes, not_targetlev, usetargetNA, dd, target, keys, usekeysNA, exclude_keys,
# #                 exclude_targetlevs, exclude_keylevs, denom_lim, exclude_ov_denom_lim):
# #     total = xx.values.sum()
# #     Nout = np.zeros(6)
# #     Nout_names = ["excluded target", "missing target", "missing in keys", "set to exclude", "over denom_lim", "remaining"]
# #
# #     # Drop all target records with not_targetlev
# #     if not_targetlev and not_targetlev != "":
# #         Nout[0] = xx.loc[xx.index.isin(not_targetlev), :].values.sum()
# #         xx.loc[xx.index.isin(not_targetlev), :] = 0
# #
# #     # Items excluded for missing values
# #     if not usetargetNA and (dd[target] == "Missing").any():
# #         Nout[2] = xx.loc[xx.index == "Missing", :].values.sum()
# #         xx.loc[xx.index == "Missing", :] = 0
# #
# #     for i, key in enumerate(keys):
# #         if not usekeysNA[i]:  # Do not use NA values for ith key
# #             key_levs = xx.columns
# #             drop_d = key_levs.str.split(" | ").str[i] == "Missing"
# #             Nout[3] += xx.loc[:, drop_d].values.sum()
# #             xx.loc[:, drop_d] = 0
# #
# #     # Remove any excluded two-way combinations
# #     if exclude_keys:
# #         if not all(ex in dd[target].cat.categories for ex in exclude_targetlevs):
# #             raise ValueError(f"exclude_targetlevs must be one of the levels of {target}")
# #
# #         for i, exclude_key in enumerate(exclude_keys):
# #             vout = [j for j, level in enumerate(xx.columns) if level == exclude_targetlevs[i]]
# #             klev = dd[exclude_key].cat.categories.tolist()
# #             if not all(ex in klev for ex in exclude_keylevs[i]):
# #                 raise ValueError(f"exclude_keylevs position {i} must be one of the levels of {keys[i]}")
# #             kind = [j for j, key in enumerate(keys) if key == exclude_key]
# #             wordk = xx.columns.str.split(" | ").str[kind[0]]
# #             kout = [j for j, word in enumerate(wordk) if word == exclude_keylevs[i]]
# #             Nout[4] += xx.iloc[vout, kout].values.sum()
# #             xx.iloc[np.ix_(vout, kout)] = 0
# #
# #     # Exclude if over denom_lim
# #     if exclude_ov_denom_lim:
# #         Nout[5] = xx[xx > denom_lim].values.sum()
# #         xx[xx > denom_lim] = 0
# #
# #     Nout[6] = total - Nout[:5].sum()
# #
# #     # Copy Nout into rows of Nexcludes
# #     Nexcludes[col, :] = Nout
# #
# #     return xx, Nexcludes
# import numpy as np
#
#
# def tab_exclude(xx, col, Nexcludes, not_targetlev, usetargetNA, dd, target, keys, usekeysNA, exclude_keys,
#                 exclude_targetlevs, exclude_keylevs, denom_lim, exclude_ov_denom_lim):
#     total = xx.values.sum()
#     Nout = np.zeros(6)
#
#     # Ensure not_targetlev is a list
#     if isinstance(not_targetlev, str):
#         not_targetlev = [not_targetlev]
#
#     # Drop all target records with not_targetlev
#     if not_targetlev and not_targetlev != [""]:
#         Nout[0] = xx.loc[xx.index.isin(not_targetlev), :].values.sum()
#         xx.loc[xx.index.isin(not_targetlev), :] = 0
#
#     # Items excluded for missing values
#     if not usetargetNA and (dd[target] == "Missing").any():
#         Nout[2] = xx.loc[xx.index == "Missing", :].values.sum()
#         xx.loc[xx.index == "Missing", :] = 0
#
#     for i, key in enumerate(keys):
#         if not usekeysNA[i]:  # Do not use NA values for ith key
#             key_levs = xx.columns
#             drop_d = key_levs.str.split(" | ").str[i] == "Missing"
#             Nout[3] += xx.loc[:, drop_d].values.sum()
#             xx.loc[:, drop_d] = 0
#
#     # Remove any excluded two-way combinations
#     if exclude_keys:
#         if not all(ex in dd[target].cat.categories for ex in exclude_targetlevs):
#             raise ValueError(f"exclude_targetlevs must be one of the levels of {target}")
#
#         for i, exclude_key in enumerate(exclude_keys):
#             vout = [j for j, level in enumerate(xx.columns) if level == exclude_targetlevs[i]]
#             klev = dd[exclude_key].cat.categories.tolist()
#             if not all(ex in klev for ex in exclude_keylevs[i]):
#                 raise ValueError(f"exclude_keylevs position {i} must be one of the levels of {keys[i]}")
#             kind = [j for j, key in enumerate(keys) if key == exclude_key]
#             wordk = xx.columns.str.split(" | ").str[kind[0]]
#             kout = [j for j, word in enumerate(wordk) if word == exclude_keylevs[i]]
#             Nout[4] += xx.iloc[vout, kout].values.sum()
#             xx.iloc[np.ix_(vout, kout)] = 0
#
#     # Exclude if over denom_lim
#     if exclude_ov_denom_lim:
#         Nout[5] = xx[xx > denom_lim].values.sum()
#         xx[xx > denom_lim] = 0
#
#     Nout[6] = total - Nout[:5].sum()
#
#     # Copy Nout into rows of Nexcludes
#     Nexcludes.iloc[col, :] = Nout
#
#     return xx, Nexcludes