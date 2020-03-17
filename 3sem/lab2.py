import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds
from numpy import nan
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

iris = load_iris()
x = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x, y)
y_model = model.predict(x)
accuracy_score(y, y_model)

x1, x2, y1, y2 = train_test_split(x, y, random_state=0, train_size=0.5)
model.fit(x1, y1)
y2_model = model.predict(x2)
accuracy_score(y2, y2_model)

y2_model = model.fit(x1, y1).predict(x2)
y1_model = model.fit(x2, y2).predict(x1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)

cross_val_score(model, x, y, cv=5)
scores = cross_val_score(model, x, y, cv=LeaveOneOut())


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def make_data(N, err=1.0, rseed=1):
    # Создаем случайные выборки данных
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


x, y = make_data(40)
X_test = np.linspace(-0.1, 1.1, 500)[:, None]
plt.scatter(x.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(x, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')
plt.show()
plt.close()

degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), x, y, 'polynomialfeatures__degree', degree, cv=7)
plt.plot(degree, np.median(train_score, 1), color='blue',
         label='training score')  # Оценка обучения
plt.plot(degree, np.median(val_score, 1), color='red',
         label='validation score')  # Оценка проверки
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')  # Степень
plt.ylabel('score')  # Оценка
plt.show()
plt.close()

x2, y2 = make_data(200)
plt.scatter(x2.ravel(), y2)
plt.show()
plt.close()
degree = np.arange(21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), x2, y2, 'polynomialfeatures__degree', degree, cv=7)
plt.plot(degree, np.median(train_score2, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')
plt.plot(degree, np.median(train_score, 1), color='blue', alpha=0.3, linestyle='dashed')
plt.plot(degree, np.median(val_score, 1), color='red', alpha=0.3, linestyle='dashed')
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree), x, y, cv=7, train_sizes=np.linspace(0.3, 1, 25))
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray', linestyle='dashed')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')  # Размерность обучения
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')
plt.show()
plt.close()

param_grid = {'polynomialfeatures__degree': np.arange(21), 'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(x, y)
print(grid.best_params_)

model = grid.best_estimator_
plt.scatter(x.ravel(), y)
lim = plt.axis()
y_test = model.fit(x, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)
plt.show()
plt.close()

data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)

sample = ['problem of evil',
          'evil queen',
          'horizon problem']
vec = CountVectorizer()
X = vec.fit_transform(sample)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
df2 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)
plt.show()
plt.close()

X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)
plt.show()
plt.close()

poly = PolynomialFeatures(degree=3, include_bias=False)
x2 = poly.fit_transform(X)
print(x2)

model = LinearRegression().fit(x2, y)
yfit = model.predict(x2)
plt.scatter(x, y)
plt.plot(x, yfit)
plt.show()
plt.close()

X = np.array([[nan, 0, 3],
              [3, 7, 9],
              [3, 5, 2],
              [4, nan, 6],
              [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
model = LinearRegression().fit(X2, y)
model.predict(X2)

model = make_pipeline(SimpleImputer(strategy='mean'), PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)  # Вышеприведенный массив X с пропущенными значениями
print(y)
print(model.predict(X))

sns.set()
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()
plt.close()

model = GaussianNB()
model.fit(X, y)
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.show()
plt.close()

plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.show()
plt.close()

yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)

data = fetch_20newsgroups()
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])
model = make_pipeline(TfidfVectorizer(decode_error='ignore'), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
plt.close()


def predict_category(s, train_p=train, model_p=model):
    pred = model_p.predict([s])
    return train_p.target_names[pred[0]]


print(predict_category('sending a payload to the ISS'))

categories = ["business", "entertainment", "politics", "sport", "tech"]
train = ds.load_files("bbc", categories=categories)
test = ds.load_files("bbc", categories=categories)
model = make_pipeline(TfidfVectorizer(decode_error='ignore'), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(data)


def predict_category_bbc(s, train_p=train, model_p=model):
    pred = model_p.predict([s])
    return train_p.target_names[pred[0]]


print(predict_category_bbc("The firm"))
print(predict_category_bbc("President"))

