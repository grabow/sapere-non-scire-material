"""Handgeschriebene Lehrbuchfunktionen fuer die Uebertragbarkeitsstudie.

Jede Funktion wird mit dem Eingabebereich je Parameter angegeben.
"""

CASES = [
    ("triangle", {"a": (0, 15), "b": (0, 15), "c": (0, 15)}, '''
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return 0
    if a + b <= c or a + c <= b or b + c <= a:
        return 0
    if a == b and b == c:
        return 3
    if a == b or b == c or a == c:
        return 2
    return 1
'''),
    ("leap_year", {"y": (1, 3000)}, '''
def leap_year(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    if y % 4 == 0:
        return True
    return False
'''),
    ("days_in_month", {"m": (0, 13), "y": (1900, 2100)}, '''
def days_in_month(m, y):
    if m < 1 or m > 12:
        return -1
    if m == 2:
        if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
            return 29
        return 28
    if m == 4 or m == 6 or m == 9 or m == 11:
        return 30
    return 31
'''),
    ("grade", {"points": (-10, 110)}, '''
def grade(points):
    if points < 0 or points > 100:
        return -1
    if points >= 90:
        return 1
    elif points >= 80:
        return 2
    elif points >= 65:
        return 3
    elif points >= 50:
        return 4
    return 5
'''),
    ("fizzbuzz", {"n": (1, 100)}, '''
def fizzbuzz(n):
    if n % 15 == 0:
        return 3
    elif n % 5 == 0:
        return 2
    elif n % 3 == 0:
        return 1
    return 0
'''),
    ("gcd", {"a": (-100, 100), "b": (-100, 100)}, '''
def gcd(a, b):
    if a < 0:
        a = -a
    if b < 0:
        b = -b
    while b != 0:
        a, b = b, a % b
    return a
'''),
    ("is_prime", {"n": (0, 200)}, '''
def is_prime(n):
    if n < 2:
        return False
    for d in range(2, 15):
        if d * d > n:
            return True
        if n % d == 0:
            return False
    return True
'''),
    ("quadrant", {"x": (-10, 10), "y": (-10, 10)}, '''
def quadrant(x, y):
    if x == 0 or y == 0:
        return 0
    if x > 0:
        if y > 0:
            return 1
        return 4
    if y > 0:
        return 2
    return 3
'''),
    ("ticket_price", {"age": (0, 100), "weekday": (0, 6), "student": (0, 1)}, '''
def ticket_price(age, weekday, student):
    price = 12
    if age < 6:
        return 0
    if age < 18 or student == 1:
        price = price - 4
    elif age >= 65:
        price = price - 3
    if weekday == 2:
        price = price - 2
    if weekday >= 5 and age >= 18:
        price = price + 2
    return price
'''),
    ("tax_class", {"income": (0, 300000)}, '''
def tax_class(income):
    if income <= 11000:
        return 0
    elif income <= 30000:
        return 1
    elif income <= 60000:
        return 2
    elif income <= 250000:
        return 3
    return 4
'''),
    ("collatz_steps", {"n": (1, 50)}, '''
def collatz_steps(n):
    steps = 0
    while n != 1 and steps < 20:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps = steps + 1
    return steps
'''),
    ("max3", {"a": (-100, 100), "b": (-100, 100), "c": (-100, 100)}, '''
def max3(a, b, c):
    m = a
    if b > m:
        m = b
    if c > m:
        m = c
    return m
'''),
    ("digit_sum", {"n": (-9999, 9999)}, '''
def digit_sum(n):
    if n < 0:
        n = -n
    s = 0
    while n > 0:
        s = s + n % 10
        n = n // 10
    return s
'''),
]
