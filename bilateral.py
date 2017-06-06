import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


bilateral = pd.read_csv('data/bilateralmigrationmatrix20130.csv', index_col='From country', thousands=',')

# histogram of migrants
nonzero_values = bilateral.values.flatten()
nonzero_values = nonzero_values[nonzero_values != 0]
plt.hist(np.log(nonzero_values), bins=100)
plt.ylabel('occurences')
plt.xlabel('migrants - log scale')
plt.show()

# most emigrated to
print bilateral.sum(axis=0).nlargest(11)
# most emigrated from 
print bilateral.sum(axis=1).nlargest(11)

# what are the countries bulgarians emigrate to?
print bilateral.loc['Bulgaria'].nlargest(11)

# here is how to plot pie charts of the 10 largest countries to emigrate to or from
# bilateral.loc['United Kingdom'].nlargest(11).tail(10).plot.pie()

# use population as predictor
wdi_data = pd.read_csv('data/wdi_data_2013.csv', index_col = 'Unnamed: 0')
population = wdi_data[wdi_data['Indicator Name'] == 'Population, total']
population.set_index('Country Name', inplace=True)

emigration_to = bilateral.sum(axis=0).to_frame('emigration_to')
emigration_from = bilateral.sum(axis=1).to_frame('emigration_from')

# plot emigration to with population 
merged = population.merge(emigration_to, left_index=True, right_index=True, how='inner')
merged = merged.merge(emigration_from, left_index=True, right_index=True, how='inner')

merged['log_population'] = np.log(merged['2013'])
merged['log_imigration'] = np.log(merged.emigration_to + 1)
merged['log_emigration'] = np.log(merged.emigration_from + 1)

log_indicators = ['log_population', 'log_emigration', 'log_imigration']

pop_plot = sns.pairplot(merged[log_indicators][merged.isnull().any(axis=1) == 0])
pop_plot.savefig('scatter-plot-population')

# use gdp per capita PPP adjusted as predictor
wdi_data = pd.read_csv('data/wdi_data_2013.csv', index_col = 'Unnamed: 0')
gdp = wdi_data[wdi_data['Indicator Name'] == 'GDP per capita, PPP (current international $)']
gdp.set_index('Country Name', inplace=True)

emigration_to = bilateral.sum(axis=0).to_frame('emigration_to')
emigration_from = bilateral.sum(axis=1).to_frame('emigration_from')

# plot emigration to with population 
merged = gdp.merge(emigration_to, left_index=True, right_index=True, how='inner')
merged = merged.merge(emigration_from, left_index=True, right_index=True, how='inner')

merged['log_gdp'] = np.log(merged['2013'])
merged['log_emigration_to'] = np.log(merged.emigration_to + 1)
merged['log_emigration_from'] = np.log(merged.emigration_from + 1)

log_indicators = ['log_gdp', 'log_emigration_to', 'log_emigration_from']

pop_plot = sns.pairplot(merged[log_indicators][merged.isnull().any(axis=1) == 0])
pop_plot.savefig('scatter-plot-gdp')

size_corrected = np.log(bilateral.values+1)
size_corrected -= size_corrected.mean(axis=0)
size_corrected = size_corrected.T
size_corrected -= size_corrected.mean(axis=0)
