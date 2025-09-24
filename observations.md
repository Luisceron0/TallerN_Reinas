# Observaciones de Experimentos con Algoritmo Genético para N-Reinas

Este documento resume las observaciones obtenidas tras ejecutar experimentos con diferentes valores de N (tamaño del tablero), tasas de mutación y tamaños de población.

---

## Configuraciones de Experimentos

- Valores de N: 4, 6, 8, 10, 12
- Tamaños de población (mu): 20, 50, 100
- Tasas de mutación: 0.01, 0.05, 0.1, 0.15, 0.2, 0.3
- Corridas por configuración: 5
- Generaciones máximas: 500

---

## Resultados Generales

- El algoritmo logra encontrar soluciones para todos los valores de N probados, aunque la dificultad y el tiempo para encontrar la solución aumentan con N.
- El tamaño de población influye en la velocidad de convergencia: poblaciones mayores tienden a encontrar soluciones en menos generaciones promedio.
- La tasa de mutación tiene un efecto no lineal: tasas muy bajas o muy altas pueden ralentizar la convergencia, mientras que tasas intermedias (0.1 a 0.2) suelen ser más efectivas.

---

## Observaciones Específicas

### Efecto del Tamaño de Población (mu)

- Para N pequeños (4, 6), incluso poblaciones pequeñas (20) encuentran soluciones rápidamente.
- Para N mayores (10, 12), poblaciones de 100 muestran una mejora significativa en generaciones promedio para solución.
- Gráfico: `results_batch/plots/generations_vs_population.png`

### Efecto de la Tasa de Mutación

- Tasas muy bajas (0.01) tienden a estancar la evolución, con pocas soluciones encontradas.
- Tasas muy altas (0.3) pueden introducir demasiada variabilidad, dificultando la convergencia.
- Tasas intermedias (0.1 a 0.2) balancean exploración y explotación, logrando mejores resultados.
- Gráfico: `results_batch/plots/generations_vs_mutation.png`

---

## Conclusiones

- Para problemas de N-Reinas con N moderados a grandes, es recomendable usar poblaciones grandes y tasas de mutación moderadas.
- Los experimentos muestran la importancia de ajustar parámetros para optimizar el rendimiento del algoritmo genético.
- Los gráficos generados permiten visualizar claramente estas tendencias y facilitan la toma de decisiones para futuras configuraciones.

---

## Archivos de Resultados

- Resultados CSV: `results_batch/batch_results.csv`
- Gráficos: `results_batch/plots/`
