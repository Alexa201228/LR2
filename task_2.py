import scipy.optimize as opt
import sympy as sp

lambda1 = sp.symbols('lambda1')
lambda2 = sp.symbols('lambda2')


def f(x):
    return x[0]**2 + 19 * x[0] + 9 * x[1]**2 + 9 * x[1]


def g1(x):
    return 2 * x[0] + 8 * x[1] - 13


def g2(x):
    return 4 * x[0] + 7 * x[1] - 5


def findMin(f, initialGuess, cons):
    result = opt.minimize(f, initialGuess, constraints=cons)

    print("--->Начально приближение: ", initialGuess)
    print("--->f* = ", result.fun)
    print("--->x* = ", result.x)

    return result


def KKT(L):
    x = sp.symbols('x1 x2')

    dL_dx1 = sp.diff(L, x[0])
    dL_dx2 = sp.diff(L, x[1])
    dL_dlambda1 = sp.diff(L, lambda1)
    dL_dlambda2 = sp.diff(L, lambda2)

    print("--->L = ", L)
    print("--->dL_dx1: ", dL_dx1)
    print("--->dL_dx2: ", dL_dx2)
    print("--->dL_dlambda1 = ", dL_dlambda1)
    print("--->dL_dlambda2 = ", dL_dlambda2)

    return sp.solve([dL_dx1, dL_dx2, dL_dlambda1, dL_dlambda2], dict=True)


if __name__ == "__main__":

    x = sp.symbols('x1 x2')

    # НАЧАЛЬНЫЕ УСЛОВИЯ #1

    startApproximation = [0, 0]
    print("Поиск минимума[1]:")
    opt1 = findMin(f, startApproximation, {'type': 'eq', 'fun': g2})

    print("Проверка условий Куна-Таккера")
    kkt = KKT(f(x) + lambda1 * g2(x))
    print("--->Решение системы: ", kkt)

    # НАЧАЛЬНЫЕ УСЛОВИЯ #2
    startApproximation = [0, 0]
    print("Поиск минимума[2]:")
    opt2 = findMin(f, startApproximation, ({'type': 'eq', 'fun': g1}, {'type': 'eq', 'fun': g2}))

    print("Проверка условий Куна-Таккера")
    kkt = KKT(f(x) + lambda1 * g1(x) + lambda2 * g2(x))
    print("--->Решение системы: ", kkt[0])

    # НАЧАЛЬНЫЕ УСЛОВИЯ #3
    startApproximation = [0, 0]
    print("Поиск минимума[3]:")
    opt3 = findMin(f, startApproximation, {'type': 'eq', 'fun': g1})

    print("Проверка условий Куна-Таккера")
    kkt = KKT(f(x) + lambda1 * g1(x))
    print("--->Решение системы: ", kkt[0])