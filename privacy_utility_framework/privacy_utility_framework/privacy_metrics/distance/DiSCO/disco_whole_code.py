import pandas as pd

from privacy_utility_framework.privacy_utility_framework.privacy_metrics.distance.DiSCO.disclosure_synds import disclosure_synds
# from SynPrivUtil_Framework.synprivutil.privacy_metrics.DiSCO.synorig_compare import synorig_compare
from privacy_utility_framework.privacy_utility_framework.privacy_metrics.distance.DiSCO.synorig_compare import synorig_compare2


# Define the main disclosure function with a basic method dispatch mechanism
def disclosure(object, data, **kwargs):
    # Dispatch to the appropriate method based on the type of object
    if isinstance(object, pd.DataFrame):
        return disclosure_dataframe(object, data, **kwargs)
    elif isinstance(object, list):
        return disclosure_dataframe(object, data, **kwargs)
    else:
        raise ValueError(f"No disclosure method associated with class {type(object).__name__}")


# Default disclosure function for unsupported types
def disclosure_default(object, **kwargs):
    raise ValueError(f"No disclosure method associated with class {type(object).__name__}")


# Disclosure function for data frames and lists
def disclosure_dataframe(object, data, cont_na=None, keys=None, target=None, print_flag=True,
                         denom_lim=5, exclude_ov_denom_lim=False, not_targetlev=None,
                         usetargetNA=True, usekeysNA=True, exclude_keys=None,
                         exclude_keylevs=None, exclude_targetlevs=None, ngroups_target=None,
                         ngroups_keys=None, thresh_1way=(50, 90), thresh_2way=(4, 80),
                         digits=2, to_print=("short",), compare_synorig=True, **kwargs):
    # Error handling for missing object
    if object is None:
        raise ValueError("Requires parameter 'object' to give name of the synthetic data.")

    # Determine the number of elements
    if isinstance(object, list) and not isinstance(object, pd.DataFrame):
        m = len(object)
    elif isinstance(object, pd.DataFrame):
        m = 1
    else:
        raise ValueError("Object must be a data frame or a list of data frames.")

    # Handle cont_na to make it a complete named list
    cna = cont_na
    cont_na = {name: [None] for name in data.columns}

    if cna is not None:
        if not isinstance(cna, dict) or any(name == "" for name in cna.keys()):
            raise ValueError("Argument 'cont_na' must be a named dictionary with names of selected variables.")
        if any(name not in data.columns for name in cna.keys()):
            raise ValueError("Names of the list cont_na must be variables in data.")
        for name, values in cna.items():
            cont_na[name] = list(set([None] + values))

    # Adjust data using synorig.compare if needed
    if compare_synorig:
        adjust_data = synorig_compare2(object if m == 1 else object[0], data, print_flag=False)
        if not adjust_data.get('unchanged', True):
            object = adjust_data['syn']
            data = adjust_data['orig']
            print("Synthetic data or original or both adjusted with synorig.compare to try to make them comparable\n")
            if m > 1:
                print("Only the first element of the list has been adjusted and will be used here\n")
                m = 1

    # Create a custom class equivalent to R's "synds" class
    class Synds:
        def __init__(self, synds, m, cont_na):
            self.synds = synds
            self.m = m
            self.cont_na = cont_na

    synds_object = Synds(synds=object, m=m, cont_na=cont_na)

    # Call the main disclosure function with the adjusted object
    ident, attrib = disclosure_synds(
        synds_object, data, keys, target=target, denom_lim=denom_lim,
        exclude_ov_denom_lim=exclude_ov_denom_lim, print_flag=print_flag, digits=digits,
        usetargetNA=usetargetNA, usekeysNA=usekeysNA, not_targetlev=not_targetlev,
        exclude_keys=exclude_keys, exclude_keylevs=exclude_keylevs,
        exclude_targetlevs=exclude_targetlevs, ngroups_target=ngroups_target,
        ngroups_keys=ngroups_keys, thresh_1way=thresh_1way, thresh_2way=thresh_2way,
        to_print=to_print, **kwargs
    )

    return ident, attrib
