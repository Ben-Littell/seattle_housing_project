import numpy as np
import matplotlib.pyplot as plt
import math
import random as rd
import pandas as pd
import func

# seaborn fivethirtyeight Solarize_Light2
plt.style.use('Solarize_Light2')

d_df = pd.read_csv('data.csv')

# CLEAN DATA
clean_df = d_df[(d_df['price'] > 0) & (d_df['price'] < 5000000) & (d_df['sqft_living'] > 0)]

# RANDOMLY SPLIT THE DATA INTO 2 SETS
# unique value_count
tr_data, te_data = func.get_random(clean_df, 50)

# plot data from the dataframe
# clean_df.plot.scatter('sqft_living', 'price')

fig, ax = plt.subplots(nrows=1, ncols=2)
clean_df.plot.scatter('sqft_living', 'price', ax=ax[0])

# clean_df.plot.scatter('sqft_lot', 'price', ax=ax[1])
# plt.show()

# tr_data[['price', 'sqft_living', 'sqft_lot']].corr()
# print(tr_data[['price', 'sqft_living', 'sqft_lot']].corr())
# print(te_data[['price', 'sqft_living', 'sqft_lot']].corr())
lin = func.regression_eqn(tr_data.sqft_living, tr_data.price)
bilin_c1, bilin_c2, y_int1 = func.regression_eqn(tr_data.sqft_living, tr_data.price, linear=False)
# print(bilin_c1, bilin_c2, y_int1)
# TRAINING SQFT LIVING TO PRICE
# func.scatter_plot(tr_data.sqft_living, tr_data.price, lin[0][0], lin[1][0], xt='sqft Living', yt='Price',
#                         title='sqft Living to Price linear')
# func.scatter_plot_bilin(tr_data.sqft_living, tr_data.price, bilin_c1, bilin_c2, y_int1, xt='sqft Living', yt='Price',
#                         title='sqft Living to Price Bi-linear')

clean_df = clean_df.sort_values(by='statezip', axis=0)

# print(clean_df.statezip.value_counts())
test = clean_df.statezip.value_counts()
zip_list = []
for val in test.index:
    if test[val] > 95:
        zip_list.append(val)
# print(zip_list)

index_dict = {}
for val in zip_list:
    index_dict[val] = []

for val in clean_df.index.values:
    if clean_df.loc[val].statezip in zip_list:
        for v in zip_list:
            if clean_df.loc[val].statezip == v:
                index_dict[v].append(val)

stats_d = func.cal_stats_column(clean_df, index_dict, ['price', 'bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot', 'yr_built'])
print(stats_d)
for val in stats_d:
    print()
    print(val)
    for i in stats_d[val]:
        print(i)
        print(stats_d[val][i])
new_df = pd.DataFrame()
new_df['bedrooms'] = clean_df.bedrooms.values
new_df['bathrooms'] = clean_df.bathrooms.values
new_df['floors'] = clean_df.floors.values
new_df['sqft_living'] = clean_df.sqft_living.values
new_df['price'] = clean_df.price.values
new_df['statezip'] = clean_df.statezip.values

# print(new_df)


func.search_engine(new_df)


