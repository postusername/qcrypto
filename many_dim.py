import random as rand
import numpy as np
import sympy as sp
import itertools as it
from tqdm.auto import tqdm


# Определяем модуль и размерность для матриц и полиномов
MOD = 17  # Модуль для операций по модулю
M = 4  # Размерность матриц и полиномов

# Генерация случайной аффинной матрицы
def rand_affine(n_in: int, n_out: int) -> np.ndarray:
    return np.random.randint(1, MOD, size=(n_out, n_in))  # Матрица из случайных чисел от 1 до MOD-1

# Находим модульную обратную величину числа с использованием расширенного алгоритма Евклида
def modulo_inverse(x: int, m: int = MOD):
    x = x % m
    if x == 0:
        raise ValueError("No inverse for 0.")  # Для 0 нет обратного по модулю
    # Расширенный алгоритм Евклида
    t, newt = 0, 1
    r, newr = m, x
    while newr != 0:
        q = r // newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    if t < 0:
        t += m
    return t

# Нахождение обратного полинома с использованием теоремы Лагранжа
def find_inverse_poly(poly: sp.Poly):
    r = []
    for x in range(MOD):
        r.append(poly(x) % MOD)  # Вычисляем значения полинома в точках от 0 до MOD-1

    r = np.argsort(r)  # Сортируем по значениям
    x = sp.Symbol('x')  # Переменная для полинома
    items = []
    for i in range(MOD):
        # Строим лагранжевы многочлены
        items.append(r[i] * sp.prod([sp.Poly((x - c) * modulo_inverse(i - c), x, domain=sp.GF(MOD)) for c in range(MOD) if c != i]))
    result = sp.Poly(sum(items), domain=sp.GF(MOD))
    result.simplify()  # Упрощаем полином
    return result

# Проверяем, является ли полином перестановочным
def test_permut_poly(poly: sp.Poly) -> bool:
    r = set()
    for x in range(MOD):
        v = poly(x) % MOD
        if v in r:
            return False  # Если значение повторяется, то это не перестановочный полином
        else:
            r.add(poly(x))

    if len(r) != MOD:
        return False  # Проверяем, что все значения уникальны

    inv = find_inverse_poly(poly)  # Находим обратный полином
    for x in range(MOD):
        if inv(poly(x)) % MOD != x:
            return False  # Проверяем обратимость
    return True

# Генерация всех перестановочных полиномов
def all_permut_poly() -> list:
    result = []
    for m in range(2, MOD - 2):  # Перебираем степень полиномов
        for n in range(1, m - 1):
            for a in range(1, MOD):  # Коэффициенты
                poly = sp.Poly(sp.Add(sp.Pow(sp.Symbol('x'), n), a * sp.Pow(sp.Symbol('x'), m)))
                if test_permut_poly(poly):  # Проверяем, является ли полином перестановочным
                    result.append(poly)
    return result


pp = all_permut_poly()  # Находим все перестановочные полиномы
print(f"Number of permutation polynomials for modulo {MOD}: {len(pp)}")

# Генерация случайных аффинных преобразований и полиномов
S = rand_affine(M, M)
T = rand_affine(M, M)
F = np.array(rand.sample(pp, M))  # Случайный выбор перестановочных полиномов

X = sp.symbols('x0:%d' % M, domain=sp.GF(MOD))  # Переменные x0, x1, ..., x(M-1)
s_x = np.array(X).T @ S  # Аффинное преобразование
f_x = np.array([sp.Poly(F[i](s_x[i]), domain=sp.GF(MOD)).simplify() for i in range(M)])
P = [sp.Poly(p, domain=sp.GF(MOD)).simplify() for p in f_x.T @ T]  # Композиция преобразований

# Нахождение обратного аффинного преобразования
def find_inv_affine(A: np.ndarray):
    A = np.array(A, dtype=int)
    n = A.shape[0]
    I = np.eye(n, dtype=int)  # Единичная матрица

    for i in range(n):
        # Нахождение опорного элемента
        pivot = -1
        for r in range(i, n):
            if A[r, i] % MOD != 0:
                pivot = r
                break
        if pivot == -1:
            raise ValueError("Matrix not invertible")

        # Перестановка строк
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            I[[i, pivot]] = I[[pivot, i]]

        # Нормализация строки
        inv_pivot = modulo_inverse(A[i, i])
        A[i, :] = (A[i, :] * inv_pivot) % MOD
        I[i, :] = (I[i, :] * inv_pivot) % MOD

        # Устранение элемента в других строках
        factors = A[:, i].copy()
        factors[i] = 0
        nonzero_rows = (factors != 0)
        A[nonzero_rows] = (A[nonzero_rows] - factors[nonzero_rows, None] * A[i]) % MOD
        I[nonzero_rows] = (I[nonzero_rows] - factors[nonzero_rows, None] * I[i]) % MOD

    return I

P_i = find_inv_affine(T) @ [find_inverse_poly(p) for p in F] @ find_inv_affine(S)  # Обратное преобразование

# Кодирование сообщения
def encode(x: np.ndarray) -> np.ndarray:
    r = []
    replacements = {X[i]: x[i] for i in range(M)}  # Замена символов
    for i in range(M):
        r.append(P[i].eval(replacements) % MOD)  # Вычисление результата полиномов
    return np.array(r)

# Декодирование сообщения
def decode(y: np.ndarray) -> np.ndarray:
    y = y @ find_inv_affine(T) % MOD
    r = []
    for i in range(M):
        r.append(find_inverse_poly(F[i])(y[i]) % MOD)  # Обратное преобразование полиномов
    return np.array(r) @ find_inv_affine(S) % MOD

# Пример работы кодирования и декодирования
msg = np.random.randint(0, MOD, M)
print("MSG", msg)

s = msg @ S % MOD
print("S(MSG)", s)
r = [F[i](s[i]) % MOD for i in range(M)]
print("F(S(MSG))", r)
t = np.array(r) @ T % MOD
print("T(F(S(MSG)))", t)

y = encode(msg)
print("P(MSG)", y)

t_i = y @ find_inv_affine(T) % MOD
print("F(S(MSG))", t_i)
r_i = [find_inverse_poly(F[i])(t_i[i]) % MOD for i in range(M)]
print("S(MSG)", r_i)
s_i = np.array(r_i) @ find_inv_affine(S) % MOD
print("MSG", s_i)

# Полный перебор всех сообщений для проверки (для небольших размерностей)
"""
for msg in tqdm(it.product(range(MOD), repeat=M), total=MOD ** M):
    y = encode(msg)
    assert np.all(decode(y) == msg), f"Failed for {msg}"
"""
