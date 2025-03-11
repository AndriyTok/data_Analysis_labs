# Реалізувати алгоритми первісного опрацювання даних для послідовності значень,
# отриманих у результаті табуляції функції f(x) = fn(x) + random на інтервалі
# від 0 до 10 з кроком 0.1, де round випадкове число з Гаусівським розподілом із
# нульовим середнім та стандартним відхиленням 0.2. fn(x) відповідає наступному виразу:
# math.sin(0.1*x) ** 2 + math.sin(x) ** 2

# В програмі передбачити можливість зміни інтервалу табуляції, кроку табуляції
# заданої функції та параметрів шуму.

import numpy as np
import matplotlib.pyplot as plt
import math

def fn(x):
    return math.sin(0.1 * x) ** 2 + math.sin(x) ** 2

def generate_data(start, end, step, noise_mean, noise_std):
    x_values = np.arange(start, end + step, step)
    y_values = np.array([fn(x) for x in x_values])
    noise = np.random.normal(noise_mean, noise_std, len(x_values))
    y_noisy = y_values + noise
    return x_values, y_values, y_noisy

def smooth_data(y_noisy, window_size=5):
    return np.convolve(y_noisy, np.ones(window_size)/window_size, mode='same')

def plot_data(x, y_clean, y_noisy, y_smoothed, title="Табуляція функції"):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_clean, label="Сигнал без шуму", color="green")
    plt.plot(x, y_noisy, label="Зашумлений сигнал", color="blue", alpha=0.6)
    plt.plot(x, y_smoothed, label="Згладжений сигнал", color="red")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Отримання параметрів від користувача
try:
    start = float(input("Введіть початок інтервалу: "))
    end = float(input("Введіть кінець інтервалу: "))
    step = float(input("Введіть крок табуляції: "))
    noise_mean = float(input("Введіть середнє значення шуму: "))
    noise_std = float(input("Введіть стандартне відхилення шуму: "))
except ValueError:
    print("Некоректне введення! Використовуються значення за замовчуванням.")
    start, end, step = 0, 10, 0.1
    noise_mean, noise_std = 0, 0.2

# Генеруємо та обробляємо дані
x_vals, y_clean, y_noisy = generate_data(start, end, step, noise_mean, noise_std)
y_smoothed = smooth_data(y_noisy)

# Відображаємо результат
plot_data(x_vals, y_clean, y_noisy, y_smoothed)

