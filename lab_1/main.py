from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.express as px


def correalize(x, data_1, data_2_missed):
    data_2_real = [item for item in data_2_missed if item > 0]
    data_2_real_mean = sum(data_2_real) / len(data_2_real)
    new_data_1 = [data_1[i] for i in x if data_2_missed[i] > 0]
    new_data_1_mean = sum(new_data_1) / len(new_data_1)
    pearson = np.corrcoef(new_data_1, data_2_real)[0][1]
    restored_data = [0] * len(data_2_missed)
    for i, val in enumerate(data_2_missed):
        if val == 0:
            restored_data[i] = data_2_real_mean + 1 / pearson * (pearson * (data_1[i] - new_data_1_mean))
        else:
            restored_data[i] = data_2_missed[i]
    return restored_data


def linearize(x, missed_data, original_data):
    linearized_data = [0] * len(original_data)
    A = np.vstack([x, np.ones(len(x))]).T
    # рассчитываем коэффициенты методом наименьших квадратов
    a, b = np.linalg.lstsq(A, original_data, rcond=None)[0]
    for i, val in enumerate(missed_data):
        # 0 - пропущенное значение
        if val == 0:
            linearized_data[i] = i * a + b
        else:
            linearized_data[i] = missed_data[i]
    return linearized_data


def winsorize(lst):
    winsorized_data = [0] * len(lst)
    for i, val in enumerate(lst):
        # 0 - пропущенное значение
        if val == 0:
            if lst[i + 1] != 0:
                winsorized_data[i] = lst[i + 1]
            else:
                winsorized_data[i] = winsorized_data[i - 1]
        else:
            winsorized_data[i] = lst[i]
    return winsorized_data


def task1():
    n = int(input())
    lst = [str(random.randint(0, 1)) for x in range(1, n + 1)]
    i = 1
    print(lst)
    while i <= n:
        zero_counter = 0
        one_counter = 0
        zero_str = '0' * i
        one_str = '1' * i
        for index, val in enumerate(lst):
            result_str = val
            if i > 1:
                for k in range(1, i):
                    if 0 <= index + k < len(lst):
                        result_str += lst[index + k]
            if result_str == zero_str:
                zero_counter += 1
            if result_str == one_str:
                one_counter += 1
        if zero_counter == 0 and one_counter == 0:
            break
        print(f"{zero_str} = {zero_counter / (len(lst) + 1 - i) * 100}%")
        print(f"{one_str} = {one_counter / (len(lst) + 1 - i) * 100}%")
        i += 1


def task2():
    df = pd.read_csv('datasets/dataset-task-2.csv', header=None)
    x = df.index.values
    y = list(chain.from_iterable(df.values))
    mean = sum(x) / len(x)
    print(f"Математическое ожидание = {mean}")
    s2 = sum((i - mean) ** 2 for i in x) / len(x)
    print(f"Стандартное отклонение = {s2 ** 0.5}")
    A = np.vstack([x, np.ones(len(x))]).T
    # рассчитываем коэффициенты методом наименьших квадратов
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"Уравнение аппроксимирующей прямой = {a}x + {b} = y")
    plt.scatter(x, y)
    plt.plot(x, a * x + b, 'r')
    plt.show()


def task3():
    res = pd.read_csv("datasets/dataset-task-3.csv")
    pearson_corr = res.corr(method='pearson')
    fig = px.imshow(pearson_corr, text_auto=True)
    fig.show()
    original = pd.read_csv("datasets/dataset-task-3-mastercard-original.csv", header=None)
    original_data = [float(x) for x in list(chain.from_iterable(original.values))]
    missed = pd.read_csv("datasets/dataset-task-3-mastercard-missed.csv", header=None)
    missed_data = [float(x) for x in list(chain.from_iterable(missed.values))]
    original_visa = pd.read_csv("datasets/dataset-task-3-visa-original.csv", header=None)
    original_visa_data = [float(x) for x in list(chain.from_iterable(original_visa.values))]
    x = [int(x) for x in original.index.values]
    plt.scatter(x, original_data)
    plt.show()
    plt.scatter(x, winsorize(missed_data))
    plt.show()
    plt.scatter(x, linearize(x, missed_data, original_data))
    plt.show()
    plt.scatter(x, correalize(x, original_visa_data, missed_data))
    plt.show()


if __name__ == '__main__':
    task1()
    # task2()
    # task3()
