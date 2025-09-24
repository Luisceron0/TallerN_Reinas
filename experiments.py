"""
Experimentos en batch para el Algoritmo Genético de N-Reinas.
"""

import os
import time
import csv
import numpy as np
from genetic_algorithm import GeneticPermutation
from utils import max_pairs

def run_batch_experiments(replacement_mode='mu_plus_lambda'):
    """
    Ejecuta experimentos en batch para diferentes configuraciones.
    """
    folder = 'results_batch'
    os.makedirs(folder, exist_ok=True)
    # Configuraciones
    Ns = [4, 6, 8, 10, 12]
    population_sizes = [20, 50, 100]  # Añadido 100 para más variación
    mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    runs_per_config = 5  # Reducido para tiempo, con más configuraciones

    out_rows = []
    total = len(Ns) * len(population_sizes) * len(mutation_rates) * runs_per_config
    count = 0
    tstart = time.time()
    for N in Ns:
        for mu in population_sizes:
            lam = mu
            for mut in mutation_rates:
                for run in range(runs_per_config):
                    count += 1
                    # GA run
                    ga = GeneticPermutation(N=N, mu=mu, lam=lam, generations=500,
                                            mutation_prob=mut, tournament_k=3,
                                            replacement_mode=replacement_mode, seed=None)
                    found = False
                    gen_found = None
                    for g in range(500):
                        stats = ga.step()
                        if stats['best_fitness'] == max_pairs(N):
                            found = True
                            gen_found = stats['generation']
                            break
                    best_ind = ga.population[int(np.argmax(ga.aptitudes))]
                    best_fit = max(ga.aptitudes)
                    out_rows.append([N, mu, mut, run, found, gen_found if gen_found is not None else -1, best_fit, best_ind])
                    if count % 10 == 0:
                        print(f"Batch progress {count}/{total}")
    # Guardar CSV
    csvpath = os.path.join(folder, 'batch_results.csv')
    with open(csvpath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['N','mu','mutation','run','found','generation_found','best_fit','best_individual'])
        for r in out_rows:
            writer.writerow(r)
    print("Batch terminado. Resultados guardados en", csvpath)
    elapsed = time.time() - tstart
    print("Tiempo batch (s):", elapsed)
    return out_rows

def analyze_results(csvpath):
    """
    Analiza los resultados del CSV y genera gráficos.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv(csvpath)
    
    results = {}

    # Generations to solution by N and mu
    results['generations'] = {}
    for N in sorted(df['N'].unique()):
        results['generations'][N] = {}
        for mu in sorted(df['mu'].unique()):
            subset = df[(df['N'] == N) & (df['mu'] == mu) & (df['found'] == True)]
            if not subset.empty:
                avg_gen = subset['generation_found'].mean()
                results['generations'][N][mu] = avg_gen

    # Generations to solution by mutation rate
    results['mutation'] = {}
    for mut in sorted(df['mutation'].unique()):
        subset = df[(df['mutation'] == mut) & (df['found'] == True)]
        if not subset.empty:
            avg_gen = subset['generation_found'].mean()
            results['mutation'][mut] = avg_gen

    # Crear carpeta para gráficos
    folder = os.path.dirname(csvpath)
    plot_folder = os.path.join(folder, 'plots')
    os.makedirs(plot_folder, exist_ok=True)

    # Gráfico: Generaciones promedio para cada N y mu
    plt.figure(figsize=(10,6))
    for N in results['generations']:
        mus = sorted(results['generations'][N].keys())
        gens = [results['generations'][N][mu] for mu in mus]
        plt.plot(mus, gens, marker='o', label=f'N={N}')
    plt.xlabel('Tamaño de población (mu)')
    plt.ylabel('Generaciones promedio para solución')
    plt.title('Generaciones promedio para solución según N y tamaño de población')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'generations_vs_population.png'), dpi=150)
    plt.close()

    # Gráfico: Generaciones promedio para cada tasa de mutación
    plt.figure(figsize=(10,6))
    muts = sorted(results['mutation'].keys())
    gens = [results['mutation'][mut] for mut in muts]
    plt.plot(muts, gens, marker='s', color='red')
    plt.xlabel('Tasa de mutación')
    plt.ylabel('Generaciones promedio para solución')
    plt.title('Generaciones promedio para solución según tasa de mutación')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'generations_vs_mutation.png'), dpi=150)
    plt.close()

    print(f"Gráficos guardados en {plot_folder}")

    return results
