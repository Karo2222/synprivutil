import pandas as pd
from rdt import HyperTransformer

# NOTE: I am testing stuff with the different datsets in here

ht = HyperTransformer()

synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/syn_SD2011_selected_columns.csv')

ht.detect_initial_config(data=synthetic_data)
print(ht.get_config())
ht.fit(synthetic_data)
transformed_data_syn = ht.transform(synthetic_data)
print(transformed_data_syn)

transformed_data_syn.to_csv('transformed_syn_SD2011_selected_columns.csv', index=False)

ht_o = HyperTransformer()

real_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/SD2011_selected_columns.csv')

ht_o.detect_initial_config(data=real_data)
print(ht_o.get_config())
ht_o.fit(real_data)
transformed_data_orig = ht_o.transform(real_data)
print(transformed_data_orig)

transformed_data_orig.to_csv('transformed_SD2011_selected_columns.csv', index=False)