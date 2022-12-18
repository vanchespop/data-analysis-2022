import numpy as np
import matplotlib.pyplot as plt
import random


def mc_points(a, b, n):
    inner_y, outer_y, inner_x, outer_x = [], [], [], []
    gen_x = [random.uniform(0, a) for i in range(1, n + 1)]
    gen_y = [random.uniform(0, b) for i in range(1, n + 1)]
    inner_counter = 0
    for k in range(n):
        if func(gen_x[k]) >= gen_y[k]:
            inner_counter += 1
            inner_x.append(gen_x[k])
            inner_y.append(gen_y[k])
        else:
            outer_x.append(gen_x[k])
            outer_y.append(gen_y[k])
    calc_square = inner_counter / n * a * b
    return inner_x, inner_y, outer_x, outer_y, calc_square


def circle(x):
    return pow((1 - x ** 2), 0.5)


def func(x):
    return 4 * x ** 0.5


def task1():
    # число экспериментов
    n = 10000
    x = np.linspace(0, 10)
    y = func(x)
    a = max(x)
    b = max(y)
    # original square
    og_square = np.trapz(y, x)
    inner_x, inner_y, outer_x, outer_y, calc_square = mc_points(a, b, n)
    print(f'Площадь, вычисленная математически {og_square} \nПлощадь, вычисленная методом Монте-Карло {calc_square} \n')
    plt.scatter(inner_x, inner_y, s=3, edgecolors='r')
    plt.scatter(outer_x, outer_y, s=3, edgecolors='b')
    plt.fill_between(x, y,
                     alpha=0.3,
                     color='green',
                     linewidth=2,
                     linestyle='-'
                     )
    plt.show()


def task2():
    # число экспериментов
    n = 100000
    # число студентов в группе
    s = 30
    match_birthday_counter = 0
    for i in range(0, n):
        lst = [random.randint(1, 365) for x in range(0, n)]
        sample = [random.choice(lst) for i in range(0, s)]
        if len(set(sample)) == s - 1:
            match_birthday_counter += 1
    print(f'{match_birthday_counter / n * 100}%')


def task3():
    # число экспериментов
    n = 10000
    # счетчик попаданий в круг
    inner_counter = 0
    my_circle = plt.Circle((0, 0), 1, color='r', alpha=.3)
    plt.gca().add_patch(my_circle)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.axis("equal")
    inner_y, outer_y, inner_x, outer_x = [], [], [], []
    for i in range(0, n):
        point = [random.uniform(-1, 1), random.uniform(-1, 1)]
        x, y = point
        if (pow(x, 2) + pow(y, 2)) < 1:
            inner_counter += 1
            inner_x.append(x)
            inner_y.append(y)
            plt.scatter(x, y, s=3, edgecolors='r')
        else:
            outer_x.append(x)
            outer_y.append(y)
    print(4 * inner_counter / n)
    plt.scatter(inner_x, inner_y, s=3, edgecolors='r')
    plt.scatter(outer_x, outer_y, s=3, edgecolors='g')
    plt.show()


if __name__ == '__main__':
    # task1()
    # task2()
    task3()
