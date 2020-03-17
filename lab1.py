import numpy as np
import csv

matrix = np.zeros((3, 5), dtype='float32')
print(matrix)

ident = np.ones((10, 10), dtype=float)
print(ident)

rand = np.random.random((5, 5))
print(rand)
print(rand.ndim)
print(rand.size)
print(rand.dtype)

print(rand[1:4, 1:4])

print(rand[0].copy().reshape(5, 1))

fifty = np.array([[j for j in range(i, i + 50, 10)] for i in range(10, 250, 50)])
print(fifty)
reversed_fifty = np.array([row[::-1] for row in fifty])
print(reversed_fifty)
print(np.concatenate([fifty, reversed_fifty], axis=1))

print([row for row in fifty])
print([column for column in fifty.transpose()])

twenty = np.array([[j for j in range(i, i + 5)] for i in range(0, 20, 5)])
print(twenty)
print(np.sqrt(twenty))

five_row = np.array([1, 2, 3, 4, 5])
five_column = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
print(np.dot(five_row, five_column))

print([np.sum(row) for row in twenty])
print([np.average(column) for column in twenty.transpose()])

reader = csv.reader(open("president_heights.csv", "rt"), delimiter=",")
heights = np.array([[float(row[0]), float(row[2])] for row in list(reader)[1:]])
avg = np.average(heights[:,1])
print(np.count_nonzero(heights[:,1] < avg))

fives = np.array([[j for j in range(i, i + 5)] for i in range(0, 25, 5)])
print(fives.diagonal())
main_diagonal = fives.diagonal()
print(np.diagflat(main_diagonal))

rand = np.random.random((5, 5))
print(rand)

print(np.sort(rand, axis=1))
print(np.sort(rand, axis=0))
