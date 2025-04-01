import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import math

def fn(x):
    return math.sin(0.1 * x) ** 2 + math.sin(x) ** 2

def generate_data(start, end, step, noise_mean, noise_std):
    x_values = np.arange(start, end + step, step)
    y_values = np.array([fn(x) for x in x_values])
    noise = np.random.normal(noise_mean, noise_std, len(x_values))
    y_noisy = y_values + noise
    return x_values, y_values, y_noisy

def moving_average(y_noisy, window_size=5):
    smoothed = np.zeros_like(y_noisy)
    half_window = window_size // 2
    for i in range(len(y_noisy)):
        start = max(0, i - half_window)
        end = min(len(y_noisy), i + half_window + 1)
        smoothed[i] = sum(y_noisy[start:end]) / (end - start)
    return smoothed

def exponential_smoothing(y_noisy, alpha=0.3):
    smoothed = np.zeros_like(y_noisy)
    smoothed[0] = y_noisy[0]
    for t in range(1, len(y_noisy)):
        smoothed[t] = alpha * y_noisy[t] + (1 - alpha) * smoothed[t-1]
    return smoothed


# Параметри дослідження
start, end, step = 0, 10, 0.1
noise_mean, noise_std = 0, 0.2
window_sizes = [2, 4, 6, 8]
alphas = [0.3, 0.5, 0.7, 0.9]

# Генерація даних
x_vals, y_clean, y_noisy = generate_data(start, end, step, noise_mean, noise_std)

# Обчислення MAE для ковзного середнього
mae_moving = {}
print("Середня абсолютна похибка для методу ковзного середнього:")
for window_size in window_sizes:
    y_smooth = moving_average(y_noisy, window_size)
    mae = mean_absolute_error(y_clean[:len(y_smooth)], y_smooth)
    mae_moving[window_size] = mae
    print(f"l = {window_size}  → MAE = {mae:.5f}")

# Обчислення MAE для експоненціального згладжування
mae_exponential = {}
print("\nСередня абсолютна похибка для методу експоненціального згладжування:")
for alpha in alphas:
    y_smooth = exponential_smoothing(y_noisy, alpha)
    mae = mean_absolute_error(y_clean, y_smooth)
    mae_exponential[alpha] = mae
    print(f"α = {alpha} → MAE = {mae:.5f}")

# Побудова графіків для ковзного середнього
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, window_size in enumerate(window_sizes):
    y_smooth = moving_average(y_noisy, window_size)
    x_smooth = x_vals[:len(y_smooth)]
    axes[i].plot(x_vals, y_clean, label="Початковий сигнал", linewidth=2, color='blue')
    axes[i].plot(x_vals, y_noisy, label="Зашумлений сигнал", alpha=0.5, color='gray')
    axes[i].plot(x_smooth, y_smooth, label=f"Ковзне середнє (l={window_size})", linewidth=2, color='red')
    axes[i].set_title(f"Ковзне середнє (l={window_size})")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()

# Побудова графіків для експоненційного згладжування
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, alpha in enumerate(alphas):
    y_smooth = exponential_smoothing(y_noisy, alpha)
    axes[i].plot(x_vals, y_clean, label="Початковий сигнал", linewidth=2, color='blue')
    axes[i].plot(x_vals, y_noisy, label="Зашумлений сигнал", alpha=0.5, color='gray')
    axes[i].plot(x_vals, y_smooth, label=f"Експоненційне згладжування (α={alpha})", linewidth=2, color='green')
    axes[i].set_title(f"Експоненційне згладжування (α={alpha})")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()