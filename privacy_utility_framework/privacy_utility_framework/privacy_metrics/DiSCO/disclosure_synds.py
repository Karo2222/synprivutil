import pandas as pd
import numpy as np

from privacy_utility_framework.privacy_utility_framework.privacy_metrics.DiSCO import group_num, tab_exclude


def disclosure_synds(
        object, data, keys, target: str, print_flag=True,
        denom_lim=5, exclude_ov_denom_lim=False,
        not_targetlev=None, usetargetNA=True, usekeysNA=True,
        exclude_keys=None, exclude_keylevs=None, exclude_targetlevs=None,
        ngroups_target=None, ngroups_keys=None,
        thresh_1way=(50, 90), thresh_2way=(4, 80),
        digits=2, to_print=["short"], **kwargs):

    # Check input parameters
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a data frame")

    data = pd.DataFrame(data)  # Ensure data is a DataFrame
    if not hasattr(object, "synds"):
        raise ValueError("object must be an object of class synds")

    if isinstance(keys, (int, float)) and not all(1 <= k <= data.shape[1] for k in keys):
        raise ValueError(f"If keys are numeric they must be in range 1 to {data.shape[1]}")

    if isinstance(keys, (int, float)):
        keys = data.columns[keys - 1]  # Adjusting for 0-based indexing in Python

    if isinstance(target, (int, float)) and not all(1 <= t <= data.shape[1] for t in target):
        raise ValueError(f"If target is numeric it must be in range 1 to {data.shape[1]}")

    if isinstance(target, (int, float)):
        target = data.columns[target - 1]  # Adjusting for 0-based indexing in Python

    Norig = data.shape[0]

    if object.m == 1:
        names_syn = list(object.synds.columns)
    else:
        names_syn = list(object.synds[0].columns)

    # target must be a single variable in data and object$syn
    # keys must be a vector of variable names in data and in s
    # target must not be in keys
    if not (all(key in data.columns for key in keys) and
            all(key in names_syn for key in keys) and
            target in data.columns and
            target in names_syn):
        raise ValueError("keys and target must be variables in data and synthetic data.")

    if len(set(keys)) != len(keys):
        raise ValueError("keys cannot include duplicated values.")

    if not isinstance(target, str):
        raise ValueError("target must be a single variable.")

    if target in keys:
        raise ValueError("target cannot be in keys.")

    if 'target' in data.columns:
        raise ValueError("your data have a variable called 'target'; please rename in original and synthetic data.")

    if not isinstance(usetargetNA, bool):
        raise ValueError("usetargetNA must be a single logical value.")

    if isinstance(usekeysNA, bool):
        usekeysNA = [usekeysNA] * len(keys)

    if len(usekeysNA) != len(keys):
        raise ValueError("usekeysNA must be a logical value of same length as keys.")

    # get keys and usekeysNA in same order as in data
    oldkeys = keys
    keys = [col for col in data.columns if col in oldkeys]
    usekeysNA = [usekeysNA[oldkeys.index(key)] for key in keys]

    # Check excluded combinations
    if exclude_keys is not None:
        if not (len(exclude_keylevs) == len(exclude_keys) == len(exclude_targetlevs)):
            raise ValueError("All excludes must be the same length")

        if not all(k in keys for k in exclude_keys):
            raise ValueError("exclude.keys must be the name of one of your keys")

    if denom_lim is not None:
        if not (isinstance(denom_lim, int) and denom_lim > 0):
            raise ValueError("denom_lim must be an integer > 0")

    if ngroups_target is not None:
        if not isinstance(ngroups_target, int):
            raise ValueError("ngroups_target must be a single value")

        if ngroups_target == 1:
            raise ValueError("Target ngroups cannot be set to 1")

    if ngroups_keys is not None:
        if isinstance(ngroups_keys, int):
            ngroups_keys = [ngroups_keys] * len(keys)

        if len(ngroups_keys) != len(keys):
            raise ValueError("ngroups_keys must be a vector of length 1 or the same as keys")

        if any(n == 1 for n in ngroups_keys):
            raise ValueError("Elements of ngroups cannot be set to 1")

    # Define output items
    m = object.m
    allCAPs = np.full((m, 5), np.nan)
    attrib = np.full((m, 8), np.nan)
    ident = np.full((m, 4), np.nan)

    # Set column names for the matrices
    allCAPs_cols = ["baseCAPd", "CAPd", "CAPs", "DCAP", "TCAP"]
    attrib_cols = ["Dorig", "Dsyn", "iS", "DiS", "DiSCO", "DiSDiO", "max_denom", "mean_denom"]
    ident_cols = ["UiO", "UiS", "UiOiS", "repU"]

    allCAPs = pd.DataFrame(allCAPs, columns=allCAPs_cols)
    attrib = pd.DataFrame(attrib, columns=attrib_cols)
    ident = pd.DataFrame(ident, columns=ident_cols)

    Nexclusions = [None] * m
    check_1way = [None] * m
    check_2way = [None] * m

    # Restrict data sets to targets and keys
    syndata = [object.synds] if m == 1 else object.synds
    dd = data.copy()
    # print(f"DD FIRST {dd}")
    dd["target"] = dd[target]

    dd = dd[["target"] + keys]
    # print(f"DD SECOND {dd}")
    # print(syndata)
    # print(f'M was: {m}')
    for jj in range(m):
        syndata[jj]["target"] = syndata[jj][target]
        syndata[jj] = syndata[jj][["target"] + keys]

    # Get cont.na parameters for stratified synthesis
    # No strata.syn attrib
    #cna = object.cont_na.iloc[0, :] if object.strata_syn is not None else object.cont_na
    cna = object.cont_na
    # print(f'TARGET {target}')
    # print(f'KEYS {keys}')
    # Assuming cna is a dictionary and target & keys are lists of relevant keys
    # Get the corresponding elements from cna dictionary
    cna = {k: object.cont_na[k] for k in [target] + keys if k in object.cont_na}

# Convert the dictionary into a DataFrame for easier manipulation
    cna = pd.DataFrame.from_dict(cna)

    #print(f'CNA {cna}')
    # Loop through cna
    for i in range(len(cna.columns)):
        nm = cna.columns[i]
        vals = cna[nm].dropna().unique()  # Get variables with cont.na other than missing
        if len(vals) > 0:
            for val in vals:
                n_cna = np.sum((data[nm] == val) & data[nm].notna())
                if n_cna == 0:
                    raise ValueError(f"Value {val} identified as denoting a special or missing in cont.na for {nm} is not in data.")
                elif n_cna < 10 and print_flag:
                    print(f"Warning: Only {n_cna} record(s) in data with value {val} identified as denoting a missing value in cont.na for {nm}")

    # print(cna.items())
    # for i, (nm, vals) in enumerate(cna.items()):
    #     vals = vals.dropna().unique()
    #
    #     if len(vals) > 0:
    #         for val in vals:
    #             n_cna = (data[nm] == val).sum()
    #             if n_cna == 0:
    #                 raise ValueError(f"Value {val} identified as denoting a special or missing in cont.na for {nm} is not in data.")
    #             elif n_cna < 10 and print_flag:
    #                 print(f"Warning: Only {n_cna} record(s) in data with value {val} identified as denoting a missing value in cont.na for {nm}")

    # Group any continuous variables if ngroups is not NULL
    if ngroups_target is None:
        ngroups_target = 0
    if ngroups_keys is None:
        ngroups_keys = 0
    if isinstance(ngroups_keys, int):
        ngroups_keys = [ngroups_keys] * len(keys)

    ngroups = [ngroups_target] + ngroups_keys
    if print_flag and any(ng > 0 and not pd.api.types.is_numeric_dtype(dd[col]) for ng, col in zip(ngroups, dd.columns)):
        print(f"\n\nWith target {target} variable(s) you have asked to group are not numeric:")
        print(f"{[col for ng, col in zip(ngroups, dd.columns) if ng > 0 and not pd.api.types.is_numeric_dtype(dd[col])]} no grouping done for them.\n")

    if any(ngroups):
        togroup = [i for i, (ng, col) in enumerate(zip(ngroups, dd.columns)) if ng > 0 and pd.api.types.is_numeric_dtype(dd[col])]

        for i in togroup:
            syn0 = np.concatenate([syndata[j].iloc[:, i].values for j in range(m)])  # All synthetic values for ith var
            # print(f'i was {i}, togroup: {togroup}')
            # print(f'CNA')
            # print(cna)
            # print(f'CNA test: {cna.iloc[:, i]}')
            # print('NGROUPS')
            # print(ngroups)
            # print('NGROUPS KEYS')
            # print(ngroups_keys)
            # print('NGROUPS TARGET')
            # print(ngroups_target)
            for j in range(m):
                grpd = group_num(dd.iloc[:, i], syndata[j].iloc[:, i], syn0, ngroups[i], cont_na=cna.iloc[:, i] if cna is not None else None)

                if len(np.unique(grpd[0])) < 3:
                    grpd = group_num(dd.iloc[:, i], syndata[j].iloc[:, i], syn0, ngroups[i], style="equal", cont_na=cna.iloc[:, i] if cna is not None else None)
                    if len(np.unique(grpd[0])) < 3:
                        print(f"Only {len(np.unique(grpd[0]))} groups produced for {dd.columns[i]} even after changing method.\n")
                    elif print_flag:
                        print(f"Grouping changed from 'quantile' to 'equal' in function group_num for {dd.columns[i]} because only {len(np.unique(grpd[0]))} groups produced\n")

                syndata[j].iloc[:, i] = grpd[1]
            dd.iloc[:, i] = grpd[0]

            if print_flag:
                if i == 0:
                    print(f"Numeric values of {dd.columns[i]}, target grouped into {len(np.unique(dd.iloc[:, i]))} groups with levels {np.unique(dd.iloc[:, i])}\n")
                else:
                    print(f"Numeric values of key {dd.columns[i]} grouped into {len(np.unique(dd.iloc[:, i]))} groups with levels {np.unique(dd.iloc[:, i])}\n")

    # if print_flag and any(ng > 0 and not np.issubdtype(dd[col].dtype, np.number) for ng, col in zip(ngroups, dd.columns)):
    #     print(f"With target {target}, variable(s) you have asked to group are not numeric: "
    #           f"{[col for ng, col in zip(ngroups, dd.columns) if ng > 0 and not np.issubdtype(dd[col].dtype, np.number)]}")
    #
    # if any(ng > 0 for ng in ngroups):
    #     togroup = [i for i, (ng, col) in enumerate(zip(ngroups, dd.columns)) if ng > 0 and np.issubdtype(dd[col].dtype, np.number)]
    #
    #     for i in togroup:
    #         syn0 = pd.concat([syndata[j][dd.columns[i]] for j in range(m)])
    #         for j in range(m):
    #             dd_col, syn_col = dd.iloc[:, i], syndata[j].iloc[:, i]
    #             # Access the relevant column in the ith row of cna
    #             cna_row = cna.iloc[i]
    #             print(f'groups: {cna_row}')
    #             #grpd = group_num(dd_col, syn_col, syn0, ngroups[i], cont_na=cna[dd.columns[i]], **kwargs)
    #             print(f'CNA ILOC:')
    #             print(cna)
    #             print(f'I: {i}')
    #             print(f'TOGROUP {togroup}')
    #             print(f'groups {cna.iloc[i] }')
    #             grpd = group_num(dd_col, syn_col, syn0, ngroups[i], cont_na=cna_row if cna is not None else None)
    #             if len(grpd[0].value_counts()) < 3:
    #                 grpd = group_num(dd_col, syn_col, syn0, ngroups[i], cont_na=cna[dd.columns[i]], style="equal")
    #
    #                 if len(grpd[0].value_counts()) < 3:
    #                     print(f"Only {len(grpd[0].value_counts())} groups produced for {dd.columns[i]} even after changing method.")
    #                 elif print_flag:
    #                     print(f"Grouping changed from 'quantile' to 'equal' for {dd.columns[i]} because only {len(grpd[0].value_counts())} groups produced.")
    #
    #             syndata[j].iloc[:, i] = grpd[1]
    #         dd.iloc[:, i] = grpd[0]
    #
    #         if print_flag:
    #             col_name = dd.columns[i]
    #             num_groups = len(grpd[0].value_counts())
    #             group_levels = list(grpd[0].value_counts().index)
    #             if i == 0:
    #                 print(f"Numeric values of target {col_name} grouped into {num_groups} groups with levels {group_levels}")

    # Convert remaining numeric values into factors
    numeric_vars = [col for col in dd.columns if pd.api.types.is_numeric_dtype(dd[col])]
    if numeric_vars:
        for col in numeric_vars:
            #print(f'NUMERIC VAR {col}')
            # dd.iloc[:,col] = dd.iloc[:,col].astype('category')
            #print(f'DD TYPE {dd[col].dtype}')
            dd[col] = dd[col].astype("category")
            #dd[col] = pd.Categorical(dd[col])
            #print(f'DD TYPE AFTER {dd[col].dtype}')
            for j in range(object.m):
                #print(f'TYPE {syndata[j][col].dtype}')
                #syndata[j].iloc[:, col] = syndata[j].iloc[:, col].astype('category')
                syndata[j][col] = syndata[j][col].astype('category')
                #syndata[j].loc[:, col] = syndata[j][col].astype(str).astype('category')
                #syndata[j].loc[:, col] = syndata[j][col].astype('category')
                #syndata[j].loc[:, col] = pd.Categorical(syndata[j][col])
                #print(f'TYPE AFTER {syndata[j][col].dtype}')

    check1 = check2 = ""  # Debugging placeholders

    # Loop over each synthesis
    for jj in range(object.m):
        if print_flag:
            print(f"-------------------Synthesis {jj + 1}--------------------")

        ss = syndata[jj].copy()

        # Replace missing values with factor value of "Missing"
        def to_missing(x):
            if not isinstance(x.dtype, pd.CategoricalDtype):
                raise ValueError(f'{x} must be a categorical type')
            x = x.astype(str)
            x[pd.isna(x)] = "Missing"
            return pd.Categorical(x)

        #print(f'DD: {dd}')
        #print(f'DD TARGET {dd["target"].dtype}')

        if dd["target"].isna().any():
            dd["target"] = to_missing(dd["target"])

        if ss["target"].isna().any():
            ss["target"] = to_missing(ss["target"])


        # Apply missing conversion to key columns
        for key in keys:
            if dd[key].isna().any():
                dd[key] = to_missing(dd[key])

            if ss[key].isna().any():
                ss[key] = to_missing(ss[key])

        Nd = len(dd)
        Ns = len(ss)

        # Create composite variable for keys
        if len(keys) > 1:
            ss['keys'] = ss[keys].apply(lambda row: ' | '.join(row.astype(str)), axis=1)
            dd['keys'] = dd[keys].apply(lambda row: ' | '.join(row.astype(str)), axis=1)
        else:
            ss['keys'] = ss[keys[0]]
            dd['keys'] = dd[keys[0]]

        # Make tables of target and keys
        NKd = dd['keys'].nunique()
        NKs = ss['keys'].nunique()

        tab_kts = pd.crosstab(ss["target"], ss['keys'])
        tab_kts.to_csv('/Users/ksi/Desktop/tab_kts_python.csv')
        tab_ktd = pd.crosstab(dd["target"], dd['keys'])
        #print(f'tab_kts {tab_kts}')

        if print_flag:
            print(f"Table for target {target} from GT alone with keys has {tab_ktd.shape[0]} rows and {tab_ktd.shape[1]} columns.")

        # Extract unique key values
        Kd = dd['keys'].unique()
        Ks = ss['keys'].unique()
        Kboth = np.intersect1d(Kd, Ks)
        Kall = np.unique(np.concatenate([Kd, Ks]))

        # Extract unique target values
        Td = dd["target"].unique()
        Ts = ss["target"].unique()
        Tboth = np.intersect1d(Td, Ts)
        Tall = np.unique(np.concatenate([Td, Ts]))

            # Augment keys tables to match
        if not np.all(np.isin(Kd, Ks)):  # Some original keys not found in synthetic data
            extraKd = Kd[~np.isin(Kd, Ks)]
            extra_tab = pd.DataFrame(0, index=tab_kts.index, columns=extraKd)
            tab_kts = pd.concat([tab_kts, extra_tab], axis=1)
            tab_kts = tab_kts.reindex(sorted(tab_kts.columns), axis=1)

        if not np.all(np.isin(Ks, Kd)):  # Extra synthetic keys not in original data
            extraKs = Ks[~np.isin(Ks, Kd)]
            extra_tab = pd.DataFrame(0, index=tab_ktd.index, columns=extraKs)
            tab_ktd = pd.concat([tab_ktd, extra_tab], axis=1)
            tab_ktd = tab_ktd.reindex(sorted(tab_ktd.columns), axis=1)

        if not np.all(np.isin(Td, Ts)):  # Some original target levels not found in synthetic data
            extraTd = Td[~np.isin(Td, Ts)]
            if tab_kts.ndim == 1:
                extra_tab = pd.DataFrame(0, index=extraTd, columns=[0])
            else:
                extra_tab = pd.DataFrame(0, index=extraTd, columns=tab_kts.columns)
            tab_kts = pd.concat([tab_kts, extra_tab], axis=0)
            tab_kts = tab_kts.reindex(sorted(tab_kts.index), axis=0)
        else:
            extraTd = None

        if not np.all(np.isin(Ts, Td)):  # Extra synthetic target levels not in original data
            extraTs = Ts[~np.isin(Ts, Td)]
            extra_tab = pd.DataFrame(0, index=extraTs, columns=tab_ktd.columns)
            tab_ktd = pd.concat([tab_ktd, extra_tab], axis=0)
            tab_ktd = tab_ktd.reindex(sorted(tab_ktd.index), axis=0)
        else:
            extraTs = None

        if print_flag:
            print(f"Table for target {target} from GT & SD with all key combinations has "
                  f"{tab_ktd.shape[0]} rows and {tab_ktd.shape[1]} columns.")

        # Calculate proportions and margins
        tab_ktd_p = tab_ktd.div(tab_ktd.sum(axis=0), axis=1).fillna(0)
        tab_kts_p = tab_kts.div(tab_kts.sum(axis=0), axis=1).fillna(0)

        # Calculating tabulation sums
        tab_kd = tab_ktd.sum(axis=0)
        tab_td = tab_ktd.sum(axis=1)
        tab_ks = tab_kts.sum(axis=0)
        tab_ts = tab_kts.sum(axis=1)

        NKall = len(tab_kd)
        NKboth = len(Kboth)
        NTd = len(Td)
        NTs = len(Ts)
        Nboth = sum(tab_kd[Kboth])
        Nd_ins = sum(tab_kd[np.isin(tab_kd.index, Ks)])

        # Preparing tables for calculating attribute disclosure measures
        did = tab_ktd.copy()
        did[tab_ktd_p != 1] = 0

        dis = tab_kts.copy()
        dis[tab_kts_p != 1] = 0

        keys_syn = tab_kts.sum(axis=0)
        tab_iS = tab_ktd.copy()
        tab_iS.loc[:, keys_syn == 0] = 0

        tab_DiS = tab_ktd.copy()
        anydis = tab_kts_p.apply(lambda x: any(x == 1), axis=0)
        tab_DiS.loc[:, ~anydis] = 0

        tab_DiSCO = tab_iS.copy()
        tab_DiSCO[tab_kts_p != 1] = 0

        tab_DiSDiO = tab_DiSCO.copy()
        tab_DiSDiO[tab_ktd_p != 1] = 0

        # Initialize Nout and Nexcludes
        # Initialize Nout with zeros
        # Initialize Nout with zeros
        Nout = np.zeros(6)

        # Create a DataFrame for Nout with the given names
        Nout = pd.DataFrame([Nout], columns=["excluded target", "missing target", "missing in keys", "set to exclude", "over denom_lim", "remaining"])

        Nexcludes = pd.DataFrame(np.zeros((8, 6)), index=["original", "synthetic", "Dorig", "Dsyn", "iS", "DiS", "DiSCO", "DiSDiO"],
                         columns=["excluded target", "missing target", "missing in keys", "set to exclude", "over denom_lim", "remaining"])

        #print(f'NOT TARGET {not_targetlev}')
        # # Apply exclusions to the tables

        def tab_exclude(xx, col, Nexcludes):
            total = xx.values.sum()

            # Drop all target records with not_targetlev
            if not_targetlev and not_targetlev != "":
                Nout.iloc[0, 0] = xx.loc[xx.index.isin([not_targetlev]), :].values.sum()
                xx.loc[xx.index.isin([not_targetlev]), :] = 0

            # Items excluded for missing values
            if not usetargetNA and (dd[target] == "Missing").any():
                Nout.iloc[0, 1] = xx.loc[xx.index == "Missing", :].values.sum()
                xx.loc[xx.index == "Missing", :] = 0

            for i, key in enumerate(keys):
                if not usekeysNA[i]:  # Do not use NA values for ith key
                    key_levs = xx.columns
                    drop_d = key_levs.str.split(" | ").str[i] == "Missing"
                    Nout.iloc[0, 2] += xx.loc[:, drop_d].values.sum()
                    xx.loc[:, drop_d] = 0
            print(f'DD {dd}')
            # Remove any excluded two-way combinations
            if exclude_keys:
                if not all(ex in dd[target].cat.categories for ex in [exclude_targetlevs]):
                    raise ValueError(f"exclude_targetlevs must be one of the levels of {target}")
                for i, exclude_key in enumerate(exclude_keys):
                    vout = [j for j, level in enumerate(xx.columns) if level == exclude_targetlevs[i]]
                    klev = dd[exclude_key].cat.categories.tolist()
                    if not all(ex in klev for ex in exclude_keylevs[i]):
                        raise ValueError(f"exclude_keylevs position {i} must be one of the levels of {keys[i]}")
                    kind = [j for j, key in enumerate(keys) if key == exclude_key]
                    wordk = xx.columns.str.split(" | ").str[kind[0]]
                    kout = [j for j, word in enumerate(wordk) if word == exclude_keylevs[i]]
                    Nout.iloc[0, 3] += xx.iloc[vout, kout].values.sum()
                    xx.iloc[np.ix_(vout, kout)] = 0

            # Exclude if over denom_lim
            if exclude_ov_denom_lim:
                Nout.iloc[0, 4] = xx[xx > denom_lim].values.sum()
                xx[xx > denom_lim] = 0

            Nout.iloc[0, 5] = total - Nout.iloc[0, :5].sum()

            # Copy Nout into rows of Nexcludes
            Nexcludes.iloc[col, :] = Nout.values

            return xx, Nexcludes

        # Example usage:
        tab_ktd, Nexcludes = tab_exclude(tab_ktd, 0, Nexcludes)
        tab_kts, Nexcludes = tab_exclude(tab_kts, 1, Nexcludes)
        did, Nexcludes = tab_exclude(did, 2, Nexcludes)
        dis, Nexcludes = tab_exclude(dis, 3, Nexcludes)
        tab_iS, Nexcludes = tab_exclude(tab_iS, 4, Nexcludes)
        tab_DiS, Nexcludes = tab_exclude(tab_DiS, 5, Nexcludes)
        tab_DiSCO, Nexcludes = tab_exclude(tab_DiSCO, 6, Nexcludes)
        tab_DiSDiO, Nexcludes = tab_exclude(tab_DiSDiO, 7, Nexcludes)
        # Identity disclosure measures calculations
        tab_ks = tab_kts.sum(axis=0)
        tab_kd = tab_ktd.sum(axis=0)

        tab_ks1 = tab_ks.copy()
        tab_ks1[tab_ks1 > 1] = 0
        tab_kd1 = tab_kd.copy()
        tab_kd1[tab_kd1 > 1] = 0
        tab_kd1_s = tab_kd1[tab_kd1.index.isin(Ks)]
        tab_ksd1 = tab_kd[(tab_ks == 1) & (tab_kd == 1)]

        UiS = tab_ks1.sum() / Ns * 100
        UiO = tab_kd1.sum() / Nd * 100
        UiOiS = tab_kd1_s.sum() / Nd * 100
        repU = tab_ksd1.sum() / Nd * 100
        #ident[jj, :] = [UiO, UiS, UiOiS, repU]

        # Attribute disclosure measures calculations
        Dorig = did.sum().sum() / Nd * 100
        Dsyn = dis.sum().sum() / Ns * 100
        iS = tab_iS.sum().sum() / Nd * 100
        DiS = tab_DiS.sum().sum() / Nd * 100
        DiSCO = tab_DiSCO.sum().sum() / Nd * 100
        print(f'DISCO: {DiSCO}')
        print(f'repU: {repU}')
        DiSDiO = tab_DiSDiO.sum().sum() / Nd * 100
        # Ensure that jj is a valid index within the DataFrame
        if 0 <= jj < ident.shape[0]:  # Check that jj is within bounds of rows
            # Assign the calculated values to the appropriate row in the DataFrame
            ident.iloc[jj, :] = [UiO, UiS, UiOiS, repU]
        else:
            print(f"Index {jj} is out of bounds for the DataFrame with {ident.shape[0]} rows.")

        if 0 <= jj < attrib.shape[0]:  # Check that jj is within bounds of rows
            # Assign values using .iloc to target the specific row
            attrib.iloc[jj, :] = [
                Dorig,
                Dsyn,
                iS,
                DiS,
                DiSCO,
                DiSDiO,
                tab_DiSCO.max().max(),
                tab_DiSCO[tab_DiSCO > 0].mean().mean(),
            ]
        else:
            print(f"Index {jj} is out of bounds for the DataFrame with {attrib.shape[0]} rows.")
        #attrib[jj, :] = [Dorig, Dsyn, iS, DiS, DiSCO, DiSDiO, tab_DiSCO.max().max(), tab_DiSCO[tab_DiSCO > 0].mean().mean()]
        # Nexclusions[jj] = Nexcludes
        print(f'IDENTITY: \n{ident}')
        print(f'ATTRIBUTES: \n{attrib}')
        print("~~~~~~~~~~~~~~~ Done ~~~~~~~~~~~~~~~")
        return ident, attrib
