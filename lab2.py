import numpy as np
import pandas as pd

powers = pd.Series([2 ** i for i in range(10)])
print(powers)

letters = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
df = pd.DataFrame({'number': letters})
print(df)
print("###")
print(df[df.__len__() - 2::-2])

squares = [i * i for i in df['number']]
df['squared'] = squares
print(df)

mat1 = pd.DataFrame([[8, 5, 3, 7, 6],
                     [0, 2, 1, 2, 9],
                     [9, 9, 7, 0, 9],
                     [7, 8, 7, 0, 0],
                     [6, 1, 9, 3, 2]])

mat2 = pd.DataFrame([[1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]])
res = mat1 / mat2
res = res.replace(np.inf, np.nan)
print(res)

for i in range(5):
    for j in range(i, 5):
        if i != j:
            res[j][i] = res[j - 1][i]
print(res)

list1 = ['Obj1', 'Obj2', 'Obj3']
list2 = ['x', 'y', 'z']
list3 = ['Scene1', 'Scene2', 'Scene3']
index = pd.MultiIndex.from_product([list3],
                                   names=['Scenes'])
columns = pd.MultiIndex.from_product([list1,
                                      list2],
                                     names=['Object_name', 'param'])

datax1 = [[9, 4, 0, 1, 9, 0, 1, 8, 9],
          [0, 8, 6, 4, 3, 0, 4, 6, 8],
          [1, 8, 4, 1, 3, 6, 5, 3, 9]]
datax = pd.DataFrame(datax1, index=index, columns=columns)
print(datax)

idx = pd.IndexSlice
slice_data = datax.loc[idx[::-1], idx[:, 'x']]
print(slice_data)

names = pd.DataFrame({'Student_id': [111, 123], 'Name': ['Иванов', 'Петров']})
subjs = pd.DataFrame({'Subj_id': [11, 22], 'Name': ['Физика', 'Математика']})
marks = pd.DataFrame({'Student_id': [111, 111, 123, 333], 'Subj_id': [11, 22, 11, 22], 'Mark': [2, 4, 5, 3]})

print(marks.merge(names, on='Student_id', how='left').merge(subjs, on='Subj_id').drop(columns=['Student_id', 'Subj_id']))
