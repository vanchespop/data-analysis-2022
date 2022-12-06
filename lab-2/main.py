from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.express as px
from sklearn import preprocessing
from sklearn.decomposition import PCA


def moving_average(lst, n):
    ma_res = [0] * (len(lst) - n)
    for i in range(0, len(ma_res)):
        ma = 0
        for k in range(0, n):
            ma += lst[i + n - k]
        ma_res[i] = ma / n
    return ma_res


def weighted_moving_average(lst, weights):
    weighted = [0] * (len(lst) - len(weights))
    weights_sum = sum(weights)
    for i in range(0, len(weighted)):
        wma = 0
        for k in range(0, len(weights)):
            wma += lst[i + len(weights) - 1 - k] * weights[k]
        weighted[i] = wma / weights_sum
    return weighted


def task1():
    df = pd.read_csv('datasets/dataset.csv')
    x = [int(x) for x in df['MASTERCARD'].index.values]
    mastercard = [float(x) for x in df['MASTERCARD'].values]
    johnson = [float(x) for x in df['JOHNSON&J'].values]
    apple = [float(x) for x in df['APPLE'].values]
    tesla = [float(x) for x in df['TESLA'].values]
    ford = [float(x) for x in df['FORD'].values]
    ma_res = moving_average(tesla, 20)
    plt.scatter(x, tesla)
    plt.plot([index for index, _ in enumerate(ma_res)], ma_res, 'r')
    plt.xlabel("Дни")
    plt.ylabel("Цена в $")
    plt.show()
    # weights = list(reversed(list((range(1, 31)))))
    # print(weights)
    # wma_res = weighted_moving_average(mastercard, weights)
    # plt.scatter(x, mastercard)
    # plt.plot([index for index, _ in enumerate(wma_res)], wma_res, 'r')
    # plt.xlabel("Дни")
    # plt.ylabel("Цена в $")
    # plt.show()


def task2():
    df = pd.read_csv('datasets/dataset.csv')
    scaled_df = preprocessing.scale(df.T)
    pca = PCA()
    pca.fit(scaled_df)
    pca_data = pca.transform(scaled_df)

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    pca_df = pd.DataFrame(pca_data, index=df.columns.tolist(), columns=labels)

    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('PCA')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))

    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

    plt.show()

if __name__ == '__main__':
    task2()
