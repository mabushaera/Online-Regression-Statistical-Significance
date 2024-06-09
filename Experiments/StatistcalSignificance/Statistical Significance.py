import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare

# Set options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# # Data from the table
# data = np.array([
#     [0.72604, 0.17228, 0.72730, 0.72702, 0.72903, 0.55234, 0.88528 ,0.89873], # experiment1
#     [0.91930, 0.18765, 0.91706, 0.90067, 0.91341, 0.83704, 0.94410 , 0.96692], # experiment2
#     [0.91397, 0.18763, 0.91525, 0.89905, 0.91873, 0.95776, 0.93874 , 0.96895], # experiment3
#     # [], # experiment4
#     [0.65904, 0.14211, 0.68412, 0.65323, 0.65438, 0.63366, 0.62747 , 0.67920], # experiment5
#     [0.59786, 0.17837, 0.85437, 0.59130, 0.60162, 0.83832, 0.46776 , 0.90239], # experiment6
#     [], # experiment 7
#     [] # experiment 8
# ])

# Data from the table
data = np.array([
    [1865.20581, 5976.67276, 1778.39859, 1961.97400, 1933.74953, 3080.20632, 541.27625 , 455.90216] # experiment1
    ,
    [2430.97513, 23760.09556, 2390.53622, 2825.94930, 2122.09717, 4715.74581, 1564.91181 , 841.70463] # experiment2
    ,
    [3282.66236, 31822.14283, 3275.53508, 3887.32562, 3218.66053, 1593.63189, 2349.97800 , 1165.10646] # experiment3
    ,
    [5550.43115, 31435.32987, 5547.42274, 5658.62652, 5429.84169, 12569.60925, 5407.04210, 3565.32212] # experiment 4
    ,
    [0.33973, 0.85781, 0.31242, 0.33924, 0.33995, 0.36409, 0.36793 , 0.31687] # expriment 5
    ,
    [0.00362, 0.00709, 0.00138, 0.00358, 0.00359, 0.00151, 0.00464 , 0.00097] # exprriment 6
    ,
    [0.88983, 0.69907, 0.89519, 0.59394, 0.57859, 0.49323, 0.50757, 0.50742] # experiment 7
    ,
    [0.05089, 0.18292, 0.23790, 0.05065, 0.05058, 0.09088, 0.07098, 0.00407] # experiment 8
])

model_names = ['SGD', 'MBGD', 'LMS', 'ORR', 'OLR', 'RLS', 'PA', 'OLR-WA']

df = pd.DataFrame(data, columns=model_names)

print(df.describe())  # <- use averages to verify if matches table

print('----------------------------')

# Rank the models for each dataset
ranked_data = df.rank(axis=1, method='average', ascending=True)  # Changed to ascending=True
print("Ranked data:")
print(ranked_data)

# Calculate average ranks
average_ranks = ranked_data.mean(axis=0)
print("\nAverage ranks:")
print(average_ranks)

# Debugging step: Verify the structure of ranked_data
print("\nRanked data shape:", ranked_data.shape)
print("Ranked data values:")
print(ranked_data.values)

# Perform the Friedman test using the ranked data
chi2, p = friedmanchisquare(*ranked_data.values.T)
print(f"\nFriedman test statistic (on ranked data): {chi2}, p-value: {p}")


myFredman = ((12*8)/(8*9)) * ((5.125**2 + 7.625**2 + 4.375**2 + 5**2 + 3.875**2 + 4.875**2 + 3.875**2 + 1.250**2) - (8 * 9**2/4))
print(myFredman)










# import pandas as pd
# import numpy as np
# from scipy.stats import friedmanchisquare
#
# # Set options to display all rows and columns
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
#
# # Data from the table
# data = np.array([
#     [128.80225, 149.30887, 222.06547, 148.67605, 126.60106, 302.53559, 233.00717, 0],
#     [633.44905, 1860.47074, 269.59388, 928.21208, 812.93583, 554.13634, 1030.98245, 0],
#     [2989.76338, 1592.97613, 1221.71952, 4797.84614, 2175.11607, 295.85915, 3795.87935, 0]
# ])
#
# model_names = ['SGD', 'MBGD', 'LMS', 'ORR', 'OLR', 'RLS', 'PA', 'OLR-WA']
#
# df = pd.DataFrame(data, columns=model_names)
#
# print(df.describe())  # <- use averages to verify if matches table
#
# print('----------------------------')
#
# # Rank the models for each dataset
# ranked_data = df.rank(axis=1, method='average', ascending=True)  # Changed to ascending=True
# print("Ranked data:")
# print(ranked_data)
#
# # Calculate average ranks
# average_ranks = ranked_data.mean(axis=0)
# print("\nAverage ranks:")
# print(average_ranks)
#
# # Debugging step: Verify the structure of ranked_data
# print("\nRanked data shape:", ranked_data.shape)
# print("Ranked data values:")
# print(ranked_data.values)
#
# # Perform the Friedman test using the ranked data
# chi2, p = friedmanchisquare(*ranked_data.values.T)
# print(f"\nFriedman test statistic: {chi2}, p-value: {p}")
#
#
# # without the ranking step:
# # Perform the Friedman test
# chi2, p = friedmanchisquare(*data.T)
# print(f"Friedman test statistic: {chi2}, p-value: {p}")