import pandas as pd

ages = pd.read_csv('state-population.csv',  # Это то, куда вы скачали файл
                   sep=',',
                   index_col=['ages', 'year', 'state']
                   )

states = pd.read_csv('state-abbrevs.csv', sep=',')

# print(ages)

# task 2
# under18 = ages.loc['under18', :]
# total = ages.loc['total', :]
#
# print("under18\n", under18)
# print("total\n", total)
#
# old = total - under18
# print("old\n", old)
#
#
# res = old \
#     .reset_index() \
#     .merge(states, left_on='state', right_on='abbreviation', how='inner', left_index=True) \
#     .drop(columns=['abbreviation', 'state_x'])
# print("result\n", res)

# task 3

ages = pd.read_csv('state-population.csv',  # Это то, куда вы скачали файл
                   sep=',',
                   index_col=['state']
                   )

states = pd.read_csv('state-abbrevs.csv', sep=',')

ages = ages[ages['ages'] == 'total'].drop(columns='ages')
first_ages = ages[ages['year'] == 1990]
res = ages[ages['year'] == 1991] - first_ages
res = res.drop(columns='year')
res.columns = ['1991']

#print(ages)
#print(res)

for i in range(1992, 2014):
    res[str(i)] = (ages[ages['year'] == i] - ages[ages['year'] == i - 1]).drop(columns='year')

print(res.reset_index())


#print(first_ages)

#print(ages)
