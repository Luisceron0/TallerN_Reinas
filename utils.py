"""
Utilidades para el problema de N-Reinas y algoritmos genéticos.
"""

def conflicts_count(individual):
    """
    Calcula el número de conflictos (pares de reinas que se atacan) en una configuración.
    """
    n = len(individual)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(individual[i] - individual[j]) == abs(i - j):
                conflicts += 1
    return conflicts

def max_pairs(n):
    """
    Número máximo de pares posibles en un tablero de N x N.
    """
    return n * (n - 1) // 2
