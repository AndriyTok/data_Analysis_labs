#знайти два найбільші числа із 10 випадкових чисел
import random

array:[int] = [random.randint(1,100) for _ in range(10)]
print(array)

def find_two_max(arr):
    max_1, max_2 = 0,0

    for num in arr:
        if num > max_1:
            max_2 = max_1
            max_1 = num
        elif num > max_2:
            max_2 = num

    return max_1, max_2


max1, max2 = find_two_max(array)
print(f"Два найбільші числа: {max1}, {max2}")