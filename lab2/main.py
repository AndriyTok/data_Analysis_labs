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

def cumulative_moving_average(y_noisy):
    smoothed = np.zeros_like(y_noisy)
    sum_values = 0
    for t in range(len(y_noisy)):
        sum_values += y_noisy[t]
        smoothed[t] = sum_values / (t + 1)
    return smoothed

def moving_average(y_noisy, window_size=8):
    smoothed = np.zeros_like(y_noisy)
    half_window = window_size // 2
    for i in range(len(y_noisy)):
        start = max(0, i - half_window)
        end = min(len(y_noisy), i + half_window + 1)
        smoothed[i] = sum(y_noisy[start:end]) / (end - start)
    return smoothed

def exponential_smoothing(y_noisy, alpha=0.5):
    smoothed = np.zeros_like(y_noisy)
    smoothed[0] = y_noisy[0]
    for t in range(1, len(y_noisy)):
        smoothed[t] = alpha * y_noisy[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

def calculate_error(y_clean, y_smoothed):
    return np.mean(np.abs(y_clean - y_smoothed))

def plot_data(x, y_clean, y_noisy, y_smoothed, title):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_clean, label="Сигнал без шуму", color="green")
    plt.plot(x, y_noisy, label="Зашумлений сигнал", color="blue", alpha=0.6)
    plt.plot(x, y_smoothed, label=title, color="red")
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
    window_size = int(input("Введіть розмір вікна для ковзного середнього: "))
    alpha = float(input("Введіть коефіцієнт згладжування для EMA (0-1): "))
except ValueError:
    print("Некоректне введення! Використовуються значення за замовчуванням.")
    start, end, step = 0, 10, 0.1
    noise_mean, noise_std = 0, 0.2
    window_size = 8
    alpha = 0.5

# Генеруємо дані
x_vals, y_clean, y_noisy = generate_data(start, end, step, noise_mean, noise_std)

# Обчислення згладжених даних
y_cma = cumulative_moving_average(y_noisy)
y_sma = moving_average(y_noisy, window_size)
y_ema = exponential_smoothing(y_noisy, alpha)

# Обчислення похибок
error_cma = calculate_error(y_clean, y_cma)
error_sma = calculate_error(y_clean, y_sma)
error_ema = calculate_error(y_clean, y_ema)

print(f"Похибка для методу оновлюваного середнього: {error_cma:5f}")
print(f"Похибка для методу середнього ковзного: {error_sma:.5f}")
print(f"Похибка для методу експоненціального згладжування: {error_ema:.5f}")

# Відображення результатів
plot_data(x_vals, y_clean, y_noisy, y_cma, "Метод оновлюваного середнього")
plot_data(x_vals, y_clean, y_noisy, y_sma, "Ковзне середнє")
plot_data(x_vals, y_clean, y_noisy, y_ema, "Експоненційне згладжування")