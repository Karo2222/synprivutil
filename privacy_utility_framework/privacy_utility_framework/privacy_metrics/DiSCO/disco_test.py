import pandas as pd

from privacy_utility_framework.privacy_utility_framework.models.transform import normalize
from privacy_utility_framework.privacy_utility_framework.privacy_metrics.DiSCO.disco_whole_code import disclosure

# real_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/diabetes.csv")
# synthetic_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/synthetic_data.csv")
#
# keys = ["Pregnancies" ,"Glucose","BloodPressure"]
# target = ["Outcome"]


# real_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/SD2011_selected_columns.csv')
# synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/syn_SD2011_selected_columns.csv')
#
# keys = ["sex", "age", "region", "placesize"]
# target = "depress"
#
# real_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/insurance.csv")
# synthetic_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/insurance_datasets/random_sample.csv")
# keys = ["age", "bmi"]
# target = "sex"

# disclosure(
#     synthetic_data,  # synthetic data object
#     real_data,  # original data set
#     cont_na=None,  # no special values for continuous variables
#     keys=["age", "sex", "region"],  # quasi-identifiers
#     target="charges",  # target variable
#     print_flag=True,  # print disclosure calculation progress
#     denom_lim=10,  # limit for large key-target groups
#     exclude_ov_denom_lim=True,  # exclude over denom_lim
#     not_targetlev="high",  # exclude 'high' level of target
#     usetargetNA=False,  # do not use NA values in target
#     usekeysNA=[True, False, True],  # use NA values in some keys
#     exclude_keys=["region"],  # exclude specific key
#     exclude_keylevs=["northwest"],  # exclude specific key level
#     exclude_targetlevs=["high"],  # exclude specific target level
#     ngroups_target=2,  # group target variable into 5 categories
#     ngroups_keys=[3, 0, 4],  # group keys into specified categories
#     thresh_1way=[100, 95],  # custom thresholds for 1-way disclosure
#     thresh_2way=[10, 85],  # custom thresholds for 2-way disclosure
#     digits=3,  # number of digits to print
#     to_print=["detailed"],  # print detailed summary
#     compare_synorig=False  # do not compare synthetic and original data
# )

# disclosure(synthetic_data,real_data,
#            keys=["age", "bmi"],
#            target="sex", print_flag=True,
#            denom_lim=5, exclude_ov_denom_lim=True,
#            not_targetlev="no", usetargetNA=True,
#            usekeysNA=True, exclude_keys=["bmi"],
#            exclude_keylevs=["female"],
#            exclude_targetlevs=["female"],
#            ngroups_target=3, ngroups_keys=[2, 0],
#            thresh_1way=(50, 90), thresh_2way=(4, 80),
#            digits=2, to_print=["short"], cont_na=None)

synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
original_datasets =["diabetes"]
#"diabetes", "cardio", "insurance"

diabetes_keys = ['Age', 'BMI', 'DiabetesPedigreeFunction', 'Glucose', 'BloodPressure']
diabetes_target = 'Outcome'

cardio_keys = ['age', 'gender', 'height', 'weight', 'cholesterol', 'gluc']
cardio_target = 'cardio'

insurance_keys = ['age', 'sex', 'region']
insurance_target = 'charges'

#DisclosureCalculator
for orig in original_datasets:
    for syn in synthetic_datasets:
        original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
        synthetic_data = pd.read_csv(f"/privacy_utility_framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
        norm_orig, norm_syn = normalize(original_data, synthetic_data)
        print(f"ORIG NORM {norm_orig}")
        # Example 1: Demographics Focus
        i, a = disclosure(norm_syn, norm_orig, keys=diabetes_keys, target=diabetes_target)
        print(f"RESULT DiSCO, repU: {orig}; {syn}")
        print(i["repU"])
        print(a["DiSCO"])

        # print(f"RESULT DiSCO, repU: {orig}; {syn}")
        # # Example 2: Health Metrics Focus
        # keys = ['bmi', 'smoker', 'charges']
        # print(disclosure(synthetic_data, original_data, keys=keys, target=insurance_target))
        #
        # print(f"RESULT DiSCO, repU: {orig}; {syn}")
        # # Example 3: Comprehensive Mix
        # keys = ['age', 'bmi', 'children', 'smoker', 'region']
        # print(disclosure(synthetic_data, original_data, keys=keys, target=insurance_target))

        # ident, attrib = disclosure(synthetic_data, original_data, keys=cardio_keys, target=cardio_target)
        # print(f"RESULT DiSCO, repU: {orig}; {syn}")
        # print(ident)
        # print(attrib)

