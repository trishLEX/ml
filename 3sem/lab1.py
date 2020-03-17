import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture as GMM
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

iris = pd.read_csv('iris.csv', delimiter=';')
iris.head()

print(iris.head())

x_iris = iris.drop('Class', axis=1)

y_iris = iris['Class']

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)
plt.show()

model = LinearRegression(fit_intercept=True)
x = x[:, np.newaxis]
print(x.shape)

model.fit(x, y)
xfit = np.linspace(-1, 11)

print("xfit: " + str(xfit))

xfit = xfit[:, np.newaxis]
print("xfit: " + str(xfit))
yfit = model.predict(xfit)
print("yfit: " + str(yfit))

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x_iris, y_iris, random_state=1)
model = GaussianNB()
model.fit(x_train, y_train)
y_model = model.predict(x_test)
accuracy_score(y_test, y_model)

model = PCA(n_components=2)
model.fit(x_iris)
x_2d = model.transform(x_iris)
iris['PCA1'] = x_2d[:, 0]
iris['PCA2'] = x_2d[:, 1]
sns.lmplot("PCA1", "PCA2", hue='Class', data=iris, fit_reg=False)
plt.show()

model = GMM(n_components=3, covariance_type='full')
model.fit(x_iris)
y_gmm = model.predict(x_iris)
iris['cluster'] = y_gmm

iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='Class', col='cluster', fit_reg=False)
plt.show()

digits = load_digits()
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')

# plt.show()
x = digits.data
y = digits.target

iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)

plt.scatter(data_projected[:, 0], data_projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
model = GaussianNB()
model.fit(x_train, y_train)
y_model = model.predict(x_test)
mat = confusion_matrix(y_test, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')  # Прогнозируемое значение
plt.ylabel('true value')  # Настоящее значение
plt.show()