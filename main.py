"""
Punto de entrada principal para el Algoritmo Evolutivo de N-Reinas.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Forzar headless
import matplotlib.pyplot as plt

from genetic_algorithm import GeneticPermutation
from utils import conflicts_count, max_pairs
from experiments import run_batch_experiments, analyze_results

if __name__ == "__main__":
    print("Ejecutando en modo headless.")
    
    # Ejecutar experimentos en batch
    print("Ejecutando experimentos en batch...")
    results = run_batch_experiments()
    
    # Analizar resultados
    csvpath = 'results_batch/batch_results.csv'
    if os.path.exists(csvpath):
        analyze_results(csvpath)
    
    # Ejemplo simple para N=6
    print("\nEjecutando ejemplo para N=6...")
    ga6 = GeneticPermutation(N=6, mu=50, lam=50, generations=200, mutation_prob=0.15, tournament_k=3, replacement_mode='mu_plus_lambda', seed=42)
    hist_b6, hist_a6 = [], []
    for g in range(200):
        stats = ga6.step()
        hist_b6.append(stats['best_fitness'])
        hist_a6.append(stats['avg_fitness'])
        print(f"Gen {stats['generation']}: Mejor {stats['best_fitness']}, Prom {stats['avg_fitness']:.2f}")
        if stats['best_fitness'] == max_pairs(6):
            print("Solución encontrada en gen", stats['generation'])
            break

    # Ejemplo simple para N=8
    print("\nEjecutando ejemplo para N=8...")
    ga8 = GeneticPermutation(N=8, mu=50, lam=50, generations=200, mutation_prob=0.15, tournament_k=3, replacement_mode='mu_plus_lambda', seed=42)
    hist_b8, hist_a8 = [], []
    for g in range(200):
        stats = ga8.step()
        hist_b8.append(stats['best_fitness'])
        hist_a8.append(stats['avg_fitness'])
        print(f"Gen {stats['generation']}: Mejor {stats['best_fitness']}, Prom {stats['avg_fitness']:.2f}")
        if stats['best_fitness'] == max_pairs(8):
            print("Solución encontrada en gen", stats['generation'])
            break

    # Guardar gráficos
    os.makedirs('results_headless', exist_ok=True)

    # Gráfico para N=6
    plt.figure(figsize=(10,5))
    gens6 = list(range(1, len(hist_b6)+1))
    plt.plot(gens6, hist_b6, 'b-o', label='Mejor')
    plt.plot(gens6, hist_a6, 'r--s', label='Promedio')
    plt.xlabel('Generaciones')
    plt.ylabel('Aptitud')
    plt.title('Evolución N=6 (headless)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results_headless/evolucion_headless_n6.png', dpi=150, bbox_inches='tight')

    # Gráfico para N=8
    plt.figure(figsize=(10,5))
    gens8 = list(range(1, len(hist_b8)+1))
    plt.plot(gens8, hist_b8, 'b-o', label='Mejor')
    plt.plot(gens8, hist_a8, 'r--s', label='Promedio')
    plt.xlabel('Generaciones')
    plt.ylabel('Aptitud')
    plt.title('Evolución N=8 (headless)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results_headless/evolucion_headless_n8.png', dpi=150, bbox_inches='tight')

    # Guardar tablero final para N=6
    best_idx6 = int(np.argmax(ga6.aptitudes))
    board6 = ga6.population[best_idx6]
    fig6 = plt.figure(figsize=(6,6))
    ax6 = fig6.add_subplot(111)
    n6 = len(board6)
    for i in range(n6):
        for j in range(n6):
            color = 'whitesmoke' if (i + j) % 2 == 0 else 'lightgray'
            ax6.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor='black', linewidth=0.3))
    for col, fila in enumerate(board6):
        ax6.text(col, fila, '♛', fontsize=36, ha='center', va='center', color='darkred')
    ax6.set_xlim(-0.5, n6 - 0.5)
    ax6.set_ylim(-0.5, n6 - 0.5)
    ax6.set_xticks(range(n6))
    ax6.set_yticks(range(n6))
    plt.title('Tablero final N=6 (headless)')
    plt.tight_layout()
    plt.savefig('results_headless/tablero_final_n6.png', dpi=150, bbox_inches='tight')

    # Guardar tablero final para N=8
    best_idx8 = int(np.argmax(ga8.aptitudes))
    board8 = ga8.population[best_idx8]
    fig8 = plt.figure(figsize=(6,6))
    ax8 = fig8.add_subplot(111)
    n8 = len(board8)
    for i in range(n8):
        for j in range(n8):
            color = 'whitesmoke' if (i + j) % 2 == 0 else 'lightgray'
            ax8.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor='black', linewidth=0.3))
    for col, fila in enumerate(board8):
        ax8.text(col, fila, '♛', fontsize=36, ha='center', va='center', color='darkred')
    ax8.set_xlim(-0.5, n8 - 0.5)
    ax8.set_ylim(-0.5, n8 - 0.5)
    ax8.set_xticks(range(n8))
    ax8.set_yticks(range(n8))
    plt.title('Tablero final N=8 (headless)')
    plt.tight_layout()
    plt.savefig('results_headless/tablero_final_n8.png', dpi=150, bbox_inches='tight')

    print("Resultados guardados en carpeta 'results_headless'.")
