from flask import Flask, render_template, request, redirect, url_for
from genetic_algorithm import GeneticPermutation
from utils import conflicts_count, max_pairs
import numpy as np

app = Flask(__name__)

# Global state
ga = None
running = False
history_best = []
history_avg = []

def init_ga(N=8, mu=50, lam=50, mutation=0.15, tournament_k=3, replacement_mode='mu_plus_lambda'):
    global ga, history_best, history_avg
    ga = GeneticPermutation(N=N, mu=mu, lam=lam, mutation_prob=mutation, tournament_k=tournament_k, replacement_mode=replacement_mode, seed=42)
    history_best = []
    history_avg = []

init_ga()

@app.route('/')
def index():
    if ga is None:
        init_ga()
    best_idx = int(np.argmax(ga.aptitudes))
    board = ga.population[best_idx]
    best_fitness = max(ga.aptitudes)
    avg_fitness = float(np.mean(ga.aptitudes))
    conflicts = conflicts_count(board)
    return render_template('index.html', N=ga.N, mu=ga.mu, lam=ga.lam, mutation=ga.mutation_prob, tournament_k=ga.tournament_k,
                           replacement_mode=ga.replacement_mode, generation=ga.generation, best_fitness=best_fitness,
                           avg_fitness=avg_fitness, conflicts=conflicts, board=board, running=running,
                           history_best=history_best, history_avg=history_avg)

@app.route('/update_params', methods=['POST'])
def update_params():
    N = int(request.form['N'])
    mu = int(request.form['mu'])
    lam = int(request.form['lam'])
    mutation = float(request.form['mutation'])
    tournament_k = int(request.form['tournament_k'])
    replacement_mode = request.form['replacement_mode']
    init_ga(N, mu, lam, mutation, tournament_k, replacement_mode)
    return redirect(url_for('index'))

@app.route('/step')
def step():
    global history_best, history_avg
    if ga:
        stats = ga.step()
        history_best.append(stats['best_fitness'])
        history_avg.append(stats['avg_fitness'])
        best_idx = int(np.argmax(ga.aptitudes))
        board = ga.population[best_idx]
        best_fitness = max(ga.aptitudes)
        avg_fitness = float(np.mean(ga.aptitudes))
        conflicts = conflicts_count(board)
        return {
            'generation': ga.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'conflicts': conflicts,
            'board': list(board)
        }
    return {}

@app.route('/reset')
def reset():
    init_ga(ga.N, ga.mu, ga.lam, ga.mutation_prob, ga.tournament_k, ga.replacement_mode)
    return redirect(url_for('index'))

@app.route('/toggle_run')
def toggle_run():
    global running
    running = not running
    return redirect(url_for('index'))

class InteractiveGAApp:
    def __init__(self):
        # Parámetros iniciales
        self.N = 8
        self.mu = 50
        self.lam = 50
        self.mutation = 0.15
        self.tournament_k = 3
        self.replacement_mode = 'mu_plus_lambda'
        self.frames_per_move = 16
        self.interval = 120  # ms per frame
        self.running = False
        self.batch_thread = None
        self.batch_running = False

        # Preparar GA
        self.ga = GeneticPermutation(N=self.N, mu=self.mu, lam=self.lam,
                                     mutation_prob=self.mutation,
                                     tournament_k=self.tournament_k,
                                     replacement_mode=self.replacement_mode, seed=42)
        best_idx = int(np.argmax(self.ga.aptitudes))
        self.current_rows = np.array(self.ga.population[best_idx], dtype=float)
        self.target_rows = self.current_rows.copy()
        self.frame_count = 0

        self.history_best = []
        self.history_avg = []

        # Crear figura y subplots (mejorado: añadir subplot para conflictos)
        self.fig = plt.figure(figsize=(16, 8))
        self.ax_board = self.fig.add_subplot(2, 3, (1, 4))  # Tablero ocupa más espacio
        self.ax_stats = self.fig.add_subplot(2, 3, 5)
        self.ax_conflicts = self.fig.add_subplot(2, 3, 6)  # Nuevo subplot para conflictos
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.28, top=0.95, hspace=0.3)

        self._draw_board()
        self.queen_texts = [self.ax_board.text(i, self.current_rows[i], '♛', fontsize=28, ha='center', va='center', color='darkred') for i in range(self.N)]

        # Stats lines
        self.line_best, = self.ax_stats.plot([], [], label='Mejor', color='blue', linewidth=2, marker='o')
        self.line_avg, = self.ax_stats.plot([], [], label='Promedio', color='orange', linewidth=2, marker='s')
        self.ax_stats.set_xlim(1, 10)
        self.ax_stats.set_ylim(0, max_pairs(self.N))
        self.ax_stats.set_title("Aptitud")
        self.ax_stats.legend()

        # Conflicts plot
        self.line_conflicts, = self.ax_conflicts.plot([], [], label='Conflictos', color='red', linewidth=2)
        self.ax_conflicts.set_xlim(1, 10)
        self.ax_conflicts.set_ylim(0, max_pairs(self.N))
        self.ax_conflicts.set_title("Conflictos")
        self.ax_conflicts.legend()

        # Widgets: TextBox y Buttons (mejorado: añadir más controles)
        axbox_N = plt.axes([0.05, 0.18, 0.08, 0.05])
        axbox_mu = plt.axes([0.15, 0.18, 0.08, 0.05])
        axbox_lam = plt.axes([0.25, 0.18, 0.08, 0.05])
        axbox_mut = plt.axes([0.35, 0.18, 0.1, 0.05])
        axbox_tk = plt.axes([0.47, 0.18, 0.08, 0.05])

        ax_start = plt.axes([0.05, 0.08, 0.12, 0.06])
        ax_step = plt.axes([0.18, 0.08, 0.10, 0.06])
        ax_reset = plt.axes([0.30, 0.08, 0.10, 0.06])
        ax_batch = plt.axes([0.42, 0.08, 0.12, 0.06])
        ax_save = plt.axes([0.56, 0.08, 0.10, 0.06])
        ax_mode = plt.axes([0.68, 0.08, 0.14, 0.06])
        ax_speed = plt.axes([0.84, 0.08, 0.12, 0.06])

        self.txtN = TextBox(axbox_N, 'N', initial=str(self.N))
        self.txtMu = TextBox(axbox_mu, 'μ', initial=str(self.mu))
        self.txtLam = TextBox(axbox_lam, 'λ', initial=str(self.lam))
        self.txtMut = TextBox(axbox_mut, 'mut', initial=str(self.mutation))
        self.txtTk = TextBox(axbox_tk, 'torneo', initial=str(self.tournament_k))

        self.btn_start = Button(ax_start, 'Start/Stop')
        self.btn_step = Button(ax_step, 'Step')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_batch = Button(ax_batch, 'Run Batch')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_mode = Button(ax_mode, f'Mode: {self.replacement_mode}')
        self.slider_speed = Slider(ax_speed, 'Speed(ms)', 50, 600, valinit=self.interval)

        # Eventos
        self.txtN.on_submit(self._on_change_params)
        self.txtMu.on_submit(self._on_change_params)
        self.txtLam.on_submit(self._on_change_params)
        self.txtMut.on_submit(self._on_change_params)
        self.txtTk.on_submit(self._on_change_params)

        self.btn_start.on_clicked(self._toggle_run)
        self.btn_step.on_clicked(self._do_step)
        self.btn_reset.on_clicked(self._reset)
        self.btn_batch.on_clicked(self._run_batch)
        self.btn_save.on_clicked(self._save_current)
        self.btn_mode.on_clicked(self._toggle_mode)
        self.slider_speed.on_changed(self._change_speed)

        # Animación
        self.ani = FuncAnimation(self.fig, self._update, interval=self.interval, blit=False)
        plt.show()

    def _draw_board(self):
        self.ax_board.clear()
        n = self.N
        for i in range(n):
            for j in range(n):
                color = 'whitesmoke' if (i + j) % 2 == 0 else 'lightgray'
                self.ax_board.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor='black', linewidth=0.3))
        self.ax_board.set_xlim(-0.5, n - 0.5)
        self.ax_board.set_ylim(-0.5, n - 0.5)
        self.ax_board.set_xticks(range(n))
        self.ax_board.set_yticks(range(n))
        self.ax_board.set_title("Tablero de N-Reinas")

    def _on_change_params(self, text):
        try:
            N = int(self.txtN.text)
            mu = int(self.txtMu.text)
            lam = int(self.txtLam.text)
            mut = float(self.txtMut.text)
            tk = int(self.txtTk.text)
            if N < 4:
                print("N mínimo recomendado >= 4")
                return
            # aplicar cambios y resetear GA
            self.N = N
            self.mu = max(2, mu)
            self.lam = max(1, lam)
            self.mutation = float(mut)
            self.tournament_k = max(2, tk)
            self._reset(None)
        except Exception as e:
            print("Parámetros inválidos:", e)

    def _toggle_mode(self, event):
        if self.replacement_mode == 'mu_plus_lambda':
            self.replacement_mode = 'mu_comma_lambda'
        else:
            self.replacement_mode = 'mu_plus_lambda'
        self.btn_mode.label.set_text(f'Mode: {self.replacement_mode}')
        # Recreate GA with new mode
        self._reset(None)

    def _change_speed(self, val):
        self.interval = float(val)
        self.ani.event_source.interval = self.interval

    def _reset(self, event):
        # Re-create GA with current params
        self.ga = GeneticPermutation(N=self.N, mu=self.mu, lam=self.lam,
                                     mutation_prob=self.mutation,
                                     tournament_k=self.tournament_k,
                                     replacement_mode=self.replacement_mode, seed=None)
        best_idx = int(np.argmax(self.ga.aptitudes))
        self.current_rows = np.array(self.ga.population[best_idx], dtype=float)
        self.target_rows = self.current_rows.copy()
        self.frame_count = 0
        self.history_best = []
        self.history_avg = []
        self._draw_board()
        # recreate queen texts for new N
        for txt in getattr(self, 'queen_texts', []):
            try:
                txt.remove()
            except Exception:
                pass
        self.queen_texts = [self.ax_board.text(i, self.current_rows[i], '♛', fontsize=28, ha='center', va='center', color='darkred') for i in range(self.N)]
        self.ax_stats.set_ylim(0, max_pairs(self.N))
        self.ax_stats.set_xlim(1, 10)
        self.ax_conflicts.set_ylim(0, max_pairs(self.N))
        self.ax_conflicts.set_xlim(1, 10)
        plt.draw()

    def _toggle_run(self, event):
        self.running = not self.running

    def _do_step(self, event):
        stats = self.ga.step()
        self.target_rows = np.array(stats['best_individual'], dtype=float)
        self.frame_count = 0
        self.history_best.append(stats['best_fitness'])
        self.history_avg.append(stats['avg_fitness'])
        self._update_stats_plot()
        # interpolate to show movement immediately if stopped
        if not self.running:
            for _ in range(self.frames_per_move):
                self._interpolate_once()
            self._apply_positions()
            plt.draw()

    def _interpolate_once(self):
        t = (self.frame_count + 1) / max(1, self.frames_per_move)
        t = min(1.0, t)
        self.current_rows = (1 - t) * self.current_rows + t * self.target_rows
        self.frame_count += 1

    def _apply_positions(self):
        for col in range(self.N):
            self.queen_texts[col].set_position((col, self.current_rows[col]))

    def _update_stats_plot(self):
        gens = list(range(1, len(self.history_best) + 1))
        self.line_best.set_data(gens, self.history_best)
        self.line_avg.set_data(gens, self.history_avg)
        conflicts = [max_pairs(self.N) - f for f in self.history_best]
        self.line_conflicts.set_data(gens, conflicts)
        if len(gens) > 0:
            self.ax_stats.set_xlim(1, max(10, len(gens)))
            step = max(1, len(gens)//10)
            self.ax_stats.set_xticks(range(1, len(gens)+1, step))
            self.ax_conflicts.set_xlim(1, max(10, len(gens)))
            self.ax_conflicts.set_xticks(range(1, len(gens)+1, step))
        self.ax_stats.relim()
        self.ax_stats.autoscale_view()
        self.ax_stats.figure.canvas.draw_idle()
        self.ax_conflicts.relim()
        self.ax_conflicts.autoscale_view()
        self.ax_conflicts.figure.canvas.draw_idle()

    def _update(self, frame):
        if self.running:
            # cuando terminamos de interpolar, ejecutar siguiente generación
            if self.frame_count >= self.frames_per_move:
                stats = self.ga.step()
                self.target_rows = np.array(stats['best_individual'], dtype=float)
                self.frame_count = 0
                self.history_best.append(stats['best_fitness'])
                self.history_avg.append(stats['avg_fitness'])
                self._update_stats_plot()
            else:
                # si todavía interpolando, avanzar
                pass
        # Avanzar interpolación si no estamos en target
        if not np.allclose(self.current_rows, self.target_rows, atol=1e-3) and self.frame_count < self.frames_per_move:
            self._interpolate_once()

        # Aplicar posiciones visuales
        self._apply_positions()

        # Actualizar título
        best_fit = max(self.ga.aptitudes) if len(self.ga.aptitudes) > 0 else 0
        avg_fit = float(np.mean(self.ga.aptitudes)) if len(self.ga.aptitudes) > 0 else 0
        conflicts = conflicts_count(self.ga.population[int(np.argmax(self.ga.aptitudes))])
        self.ax_board.set_title(f"Tablero - Gen: {self.ga.generation}, Mejor: {best_fit:.0f}, Prom: {avg_fit:.2f}, Conflictos: {conflicts}")

        return self.queen_texts + [self.line_best, self.line_avg, self.line_conflicts]

    def _save_current(self, event):
        # Guardar gráfico y tablero actuales
        timestamp = int(time.time())
        folder = 'results_interactive'
        os.makedirs(folder, exist_ok=True)
        plt.tight_layout()
        fname = os.path.join(folder, f'stats_{timestamp}.png')
        self.fig.savefig(fname, dpi=150, bbox_inches='tight')
        # guardar tablero por separado
        board_fname = os.path.join(folder, f'board_{timestamp}.png')
        self._save_board_image(self.ga.population[int(np.argmax(self.ga.aptitudes))], board_fname)
        print(f"Guardado: {fname} y {board_fname}")

    def _save_board_image(self, board, filename):
        fig = plt.figure(figsize=(6, 6), dpi=140)
        ax = fig.add_subplot(111)
        n = len(board)
        for i in range(n):
            for j in range(n):
                color = 'whitesmoke' if (i + j) % 2 == 0 else 'lightgray'
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor='black', linewidth=0.3))
        for col, fila in enumerate(board):
            ax.text(col, fila, '♛', fontsize=36, ha='center', va='center', color='darkred')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_title(f'Tablero - Aptitud {max_pairs(n) - conflicts_count(board)}/{max_pairs(n)}')
        plt.tight_layout()
        fig.savefig(filename, dpi=160, bbox_inches='tight')
        plt.close(fig)

    # --------------------
    # Batch experiments
    # --------------------
    def _run_batch(self, event):
        if self.batch_running:
            print("Batch ya está en ejecución.")
            return
        # Lanzar thread para no bloquear GUI
        self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.batch_thread.start()

    def _batch_worker(self):
        self.batch_running = True
        folder = 'results_batch'
        os.makedirs(folder, exist_ok=True)
        # Ejemplo simple: probar combinaciones pequeñas y guardar CSV
        Ns = [6, 8]
        population_sizes = [20, 50]
        mutation_rates = [0.05, 0.1, 0.2]
        runs_per_config = 5

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
                        # quick GA run (stop when found or after maxgen)
                        ga = GeneticPermutation(N=N, mu=mu, lam=lam, generations=500,
                                                mutation_prob=mut, tournament_k=3,
                                                replacement_mode=self.replacement_mode, seed=None)
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
                        elapsed = 0  # omit precise timing for brevity
                        out_rows.append([N, mu, mut, run, found, gen_found if gen_found is not None else -1, best_fit, best_ind])
                        if count % 5 == 0:
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
        self.batch_running = False

if __name__ == '__main__':
    app.run(debug=True)
