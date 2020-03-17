import numpy as np
import pandas as pd

degs = [i for i in range(0, 361, 30)]
values = np.array([[np.sin(np.deg2rad(i)), np.cos(np.deg2rad(i))] for i in degs])
res = pd.DataFrame(values.transpose(), columns=degs, index=['sin', 'cos'])
print(res)

names = pd.DataFrame({'user_id': [1, 2, 3], 'user_name': ['Иванов', 'Петров', 'Сидоров']})
groups = pd.DataFrame({'user_id': [1, 2, 3], 'group': ['gr1', 'gr2', 'g3']})
res = names.merge(groups, on='user_id', how='left').drop(columns=['user_id'])
print(res.to_string(index=False))

marks = pd.DataFrame({
    'year': [2018, 2018, 2018, 2018, 2019, 2019, 2019, 2019, 2020],
    'student': ['Иванов', 'Петров', 'Иванов', 'Петров', 'Иванов', 'Иванов', 'Петров', 'Петров', 'Иванов'],
    'subject': ['physics', 'physics', 'math', 'math', 'physics', 'math', 'physics', 'math', 'it'],
    'mark': [5, 4, 4, 3, 3, 4, 5, 5, 4]
})

result = marks[marks['year'] == 2019].drop(columns=['subject']).groupby(['student']).aggregate(np.average)
print(result)

marks2018 = marks[marks['year'] == 2018].drop(columns=['student']).groupby(['subject']).aggregate(np.average)
print(marks2018)
marks2019 = marks[marks['year'] == 2019].drop(columns=['student']).groupby(['subject']).aggregate(np.average)
print(marks2019)

print((marks2019 - marks2018).drop(columns=['year']))