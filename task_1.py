import scipy.optimize as opt
import sympy as sp

u1 = sp.symbols('u1')


def f(x):
    return 4 * x[0]**2 + 4 * x[1]**2


def g1(x):
    return 2 * x[0] + 7 * x[1] - 8


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

    print("--->L = ", L)
    print("--->dL_dx1: ", dL_dx1)
    print("--->dL_dx2: ", dL_dx2)
    print("--->dL_dlambda1 = ", dL_dlambda1)

    return sp.solve([dL_dx1, dL_dx2, dL_dlambda1], dict=True)


if __name__ == "__main__":
    x = sp.symbols('x y')

    # НАЧАЛЬНЫЕ УСЛОВИЯ
    startApproximation = [0, 0]
    print("Поиск минимума:")
    opt1 = findMin(f, startApproximation, {'type': 'eq', 'fun': g1})

    print("Проверка условий Куна-Таккера")
    kt = KT(f(x) + u1 * g1(x))
    print("--->Решение системы: ", kt)

