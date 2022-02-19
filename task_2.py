import scipy.optimize as opt
import sympy as sp

u1 = sp.symbols('u1')
u2 = sp.symbols('u2')


def f(x):
    return x[0]**2 + 19 * x[0] + 9 * x[1]**2 + 9 * x[1]


def g1(x):
    return 2 * x[0] + 8 * x[1] - 13


def g2(x):
    return 4 * x[0] + 7 * x[1] - 5


def findMin(f, initialGuess, cons):
    result = opt.minimize(f, initialGuess, constraints=cons)

    print("--->Начально приближение: ", initialGuess)
    print("--->f = ", round(result.fun, 3))
    print("--->x* = ", result.x)

    return result


def KT(L):
    x = sp.symbols('x y')

    dL_dx1 = sp.diff(L, x[0])
    dL_dx2 = sp.diff(L, x[1])
    dL_dlambda1 = sp.diff(L, u1)
    dL_dlambda2 = sp.diff(L, u2)

    print("--->L = ", L)
    print("--->dL_dx1: ", dL_dx1)
    print("--->dL_dx2: ", dL_dx2)
    print("--->dL_dlambda1 = ", dL_dlambda1)
    print("--->dL_dlambda2 = ", dL_dlambda2)

    return sp.solve([dL_dx1, dL_dx2, dL_dlambda1, dL_dlambda2], dict=True)


if __name__ == "__main__":

    x = sp.symbols('x y')

    # НАЧАЛЬНЫЕ УСЛОВИЯ #1
    startApproximation = [0, 0]
    print("Поиск минимума[1]:")
    opt1 = findMin(f, startApproximation, ({'type': 'eq', 'fun': g1}, {'type': 'eq', 'fun': g2}))
    print("Проверка условий Куна-Таккера")
    kt = KT(f(x) + u1 * g2(x))
    print("--->Решение системы: ", kt)
    # НАЧАЛЬНЫЕ УСЛОВИЯ #2

    startApproximation = [0, 0]
    print("Поиск минимума[2]:")
    opt2 = findMin(f, startApproximation, {'type': 'eq', 'fun': g2})
    print("Проверка условий Куна-Таккера")
    kt = KT(f(x) + u1 * g1(x) + u2 * g2(x))
    print("--->Решение системы: ", kt[0])

    # НАЧАЛЬНЫЕ УСЛОВИЯ #3
    startApproximation = [0, 0]
    print("Поиск минимума[3]:")
    opt3 = findMin(f, startApproximation, {'type': 'eq', 'fun': g1})

    print("Проверка условий Куна-Таккера")
    kt = KT(f(x) + u1 * g1(x))
    print("--->Решение системы: ", kt[0])
    print(f'Поиск минимума: {min(round(opt1.fun, 3), round(opt2.fun, 3), round(opt3.fun, 3))}')