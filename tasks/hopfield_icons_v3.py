"""
Interactive Hopfield network on 16x16 pixel-art icons.
Requires hopfield_icons_v2.py in the same folder. Controls are listed in the window.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from hopfield_icons_v2 import ICONS, N, RULES, SIDE, aligned_field, classify, corrupt, energy, run, to_pattern

HELP = """\
MOUSE
icon, left click   cue with it
icon, right click  store / unstore
state, left drag   paint black
state, right drag  paint white

KEYS
space  play / pause
n →    one step
enter  cue again
r      random state
c      clear state
x      mixture of 3 stored
1-4    rule
m      flip / mask
↑ ↓    amount
d      async / sync
+ -    speed

Corruption settings apply
at the next cue."""


class App:
    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.names = list(ICONS)
        self.patterns = {n: to_pattern(r) for n, r in ICONS.items()}
        self.stored = set(self.names)
        self.rule = "hebb"
        self.kind = "flip"
        self.amount = 0.25
        self.sync = False
        self.speed = 16
        self.running = False
        self.ticks = 0
        self.paint = None
        self.target = self.names[0]
        self.s = -np.ones(N)
        self.build()
        self.learn()
        self.cue(self.target)
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.tick)
        self.timer.start()

    # ---------- layout ----------

    def build(self):
        self.fig = plt.figure(figsize=(15, 9))
        if getattr(self.fig.canvas, "manager", None) is not None:
            self.fig.canvas.manager.set_window_title("Hopfield network")

        self.thumbs = {}
        step = 0.78 / len(self.names)
        for k, name in enumerate(self.names):
            ax = self.fig.add_axes([0.20 + k * step, 0.86, step * 0.9, 0.11])
            ax.imshow(self.patterns[name].reshape(SIDE, SIDE), cmap="gray_r", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            self.thumbs[ax] = name

        self.ax_s = self.fig.add_axes([0.20, 0.40, 0.32, 0.40])
        self.img_s = self.ax_s.imshow(self.s.reshape(SIDE, SIDE), cmap="gray_r", vmin=-1, vmax=1)
        self.ax_s.set_xticks([])
        self.ax_s.set_yticks([])
        self.ax_s.set_title("state", fontsize=9)

        self.ax_f = self.fig.add_axes([0.54, 0.40, 0.26, 0.40])
        self.img_f = self.ax_f.imshow(np.zeros((SIDE, SIDE)), cmap="RdBu", vmin=-1, vmax=1)
        self.ax_f.set_xticks([])
        self.ax_f.set_yticks([])
        self.ax_f.set_title("s_i (h_i - θ_i), red = will flip", fontsize=9)
        self.fig.colorbar(self.img_f, cax=self.fig.add_axes([0.81, 0.42, 0.01, 0.36]))

        self.ax_e = self.fig.add_axes([0.22, 0.08, 0.28, 0.24])
        (self.line_e,) = self.ax_e.plot([], [], color="C0", marker=".", ms=3)
        self.ax_e.set_xlabel("neuron updates")
        self.ax_e.set_ylabel("energy E")

        self.ax_m = self.fig.add_axes([0.58, 0.13, 0.40, 0.19])

        self.txt = self.fig.text(0.85, 0.80, "", va="top", fontsize=9, family="monospace")
        self.fig.text(0.85, 0.64, HELP, va="top", fontsize=7, family="monospace", color="0.3")

        rules = list(RULES)
        ax = self.fig.add_axes([0.01, 0.80, 0.14, 0.15])
        ax.set_title("rule", fontsize=9, loc="left")
        self.w_rule = RadioButtons(ax, rules, active=rules.index(self.rule))
        self.w_rule.on_clicked(self.set_rule)

        ax = self.fig.add_axes([0.01, 0.66, 0.14, 0.09])
        ax.set_title("corruption", fontsize=9, loc="left")
        self.w_kind = RadioButtons(ax, ["flip", "mask"], active=0)
        self.w_kind.on_clicked(self.set_kind)

        ax = self.fig.add_axes([0.06, 0.60, 0.09, 0.03])
        self.w_amount = Slider(ax, "amount", 0.0, 1.0, valinit=self.amount, valstep=0.05)
        self.w_amount.on_changed(self.set_amount)

        ax = self.fig.add_axes([0.01, 0.47, 0.14, 0.09])
        ax.set_title("dynamics", fontsize=9, loc="left")
        self.w_dyn = RadioButtons(ax, ["asynchronous", "synchronous"], active=0)
        self.w_dyn.on_clicked(self.set_dynamics)

        ax = self.fig.add_axes([0.06, 0.41, 0.09, 0.03])
        self.w_speed = Slider(ax, "speed", 1, 256, valinit=self.speed, valstep=1)
        self.w_speed.on_changed(self.set_speed)

        specs = [
            ("Play", self.toggle_play),
            ("Step", lambda e: self.do_step()),
            ("Cue again", lambda e: self.cue(self.target)),
            ("Random", lambda e: self.set_state(self.rng.choice([-1.0, 1.0], N))),
            ("Clear", lambda e: self.set_state(-np.ones(N))),
            ("Mixture", lambda e: self.mixture()),
            ("Stability", lambda e: self.show_stability()),
            ("Spurious", lambda e: self.show_spurious()),
            ("Capacity", lambda e: self.show_capacity()),
        ]
        self.buttons = []
        for k, (label, callback) in enumerate(specs):
            ax = self.fig.add_axes([0.01 + (k % 2) * 0.075, 0.33 - (k // 2) * 0.055, 0.07, 0.045])
            button = Button(ax, label)
            button.label.set_fontsize(8)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.play_button = self.buttons[0]

        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("button_release_event", self.on_release)
        canvas.mpl_connect("key_press_event", self.on_key)

    def style_thumbs(self):
        for ax, name in self.thumbs.items():
            on = name in self.stored
            ax.images[0].set_alpha(1.0 if on else 0.25)
            ax.set_title(name, fontsize=8, color="black" if on else "0.6")
            for spine in ax.spines.values():
                spine.set_color("C1" if name == self.target else "0.6")
                spine.set_linewidth(2.5 if name == self.target else 0.8)

    def color_bars(self):
        for bar, name in zip(self.bars, self.stored_names):
            bar.set_color("C1" if name == self.target else "C0")

    # ---------- network ----------

    def learn(self):
        self.stored_names = [n for n in self.names if n in self.stored]
        self.X = np.array([self.patterns[n] for n in self.stored_names])
        self.W, self.theta = RULES[self.rule](self.X)
        G = np.array([aligned_field(self.W, self.theta, x) for x in self.X])
        lim = float(np.abs(G).max()) or 1.0
        self.img_f.set_clim(-lim, lim)

        self.ax_m.clear()
        self.bars = self.ax_m.bar(range(len(self.stored_names)), np.zeros(len(self.stored_names)), color="C0")
        self.ax_m.axhline(0, color="0.5", lw=0.8)
        self.ax_m.set_ylim(-1.05, 1.05)
        self.ax_m.set_xticks(range(len(self.stored_names)))
        self.ax_m.set_xticklabels(self.stored_names, rotation=45, ha="right", fontsize=8)
        self.ax_m.set_ylabel("overlap m")
        self.color_bars()

        self.t = 0
        self.hist_t, self.hist_E = [], []
        self.reset_sweep()
        self.record()
        self.redraw()

    def reset_sweep(self):
        self.order = self.rng.permutation(N)
        self.pos = 0
        self.changed = False
        self.converged = False
        self.prev = []
        self.status = ""

    def record(self):
        self.hist_t.append(self.t)
        self.hist_E.append(energy(self.W, self.theta, self.s))

    def async_steps(self, k):
        for _ in range(k):
            if self.pos == N:
                if not self.changed:
                    self.converged = True
                    self.status = "fixed point"
                    return
                self.order = self.rng.permutation(N)
                self.pos = 0
                self.changed = False
            i = self.order[self.pos]
            self.pos += 1
            h = self.W[i] @ self.s - self.theta[i]
            if abs(h) > 1e-12 and np.sign(h) != self.s[i]:
                self.s[i] = -self.s[i]
                self.changed = True
            self.t += 1

    def sync_step(self):
        if self.converged:
            return
        h = self.W @ self.s - self.theta
        new = np.where(np.abs(h) > 1e-12, np.sign(h), self.s)
        self.t += N
        if np.array_equal(new, self.s):
            self.converged = True
            self.status = "fixed point"
        elif self.prev and np.array_equal(new, self.prev[-1]):
            self.status = "2-cycle"
        self.prev = [self.s.copy()]
        self.s = new

    def do_step(self):
        if self.sync:
            self.sync_step()
        else:
            self.async_steps(self.speed)
        self.record()
        self.redraw()

    def tick(self):
        if not self.running:
            return
        self.ticks += 1
        if self.sync and self.ticks % 8:
            return
        if self.converged:
            self.set_running(False)
            return
        self.do_step()

    # ---------- state changes ----------

    def set_state(self, s):
        self.s = np.array(s, dtype=float)
        self.t = 0
        self.hist_t, self.hist_E = [], []
        self.reset_sweep()
        self.record()
        self.redraw()

    def cue(self, name):
        self.target = name
        self.style_thumbs()
        self.color_bars()
        self.set_state(corrupt(self.patterns[name], self.kind, self.amount, self.rng))

    def mixture(self):
        if len(self.X) >= 3:
            self.set_state(np.sign(self.X[0] + self.X[1] + self.X[2]))

    def toggle_stored(self, name):
        if name in self.stored:
            if len(self.stored) == 1:
                return
            self.stored.remove(name)
        else:
            self.stored.add(name)
        self.style_thumbs()
        self.learn()

    def paint_at(self, event):
        if event.xdata is None or event.ydata is None:
            return
        r = int(np.clip(round(event.ydata), 0, SIDE - 1))
        c = int(np.clip(round(event.xdata), 0, SIDE - 1))
        i = r * SIDE + c
        if self.s[i] != self.paint:
            self.s[i] = self.paint
            self.reset_sweep()
            self.record()
            self.redraw()

    # ---------- widget callbacks ----------

    def set_rule(self, label):
        self.rule = label
        self.learn()

    def set_kind(self, label):
        self.kind = label
        self.redraw()

    def set_amount(self, val):
        self.amount = float(val)
        self.redraw()

    def set_dynamics(self, label):
        self.sync = label == "synchronous"
        self.reset_sweep()
        self.redraw()

    def set_speed(self, val):
        self.speed = int(val)

    def set_running(self, flag):
        self.running = flag
        self.play_button.label.set_text("Pause" if flag else "Play")
        self.redraw()

    def toggle_play(self, event=None):
        self.set_running(not self.running)

    # ---------- mouse and keyboard ----------

    def on_press(self, event):
        if event.inaxes in self.thumbs:
            name = self.thumbs[event.inaxes]
            if event.button == 1:
                self.cue(name)
            elif event.button == 3:
                self.toggle_stored(name)
        elif event.inaxes is self.ax_s and event.button in (1, 3):
            self.paint = 1.0 if event.button == 1 else -1.0
            self.paint_at(event)

    def on_motion(self, event):
        if self.paint is not None and event.inaxes is self.ax_s:
            self.paint_at(event)

    def on_release(self, event):
        self.paint = None

    def on_key(self, event):
        k = event.key
        if k == " ":
            self.toggle_play()
        elif k in ("n", "right"):
            self.do_step()
        elif k == "enter":
            self.cue(self.target)
        elif k in ("1", "2", "3", "4"):
            self.w_rule.set_active(int(k) - 1)
        elif k == "m":
            self.w_kind.set_active(1 - ["flip", "mask"].index(self.kind))
        elif k in ("up", "down"):
            self.w_amount.set_val(float(np.clip(self.amount + (0.05 if k == "up" else -0.05), 0.0, 1.0)))
        elif k == "d":
            self.w_dyn.set_active(0 if self.sync else 1)
        elif k in ("+", "="):
            self.w_speed.set_val(min(256, self.speed * 2))
        elif k == "-":
            self.w_speed.set_val(max(1, self.speed // 2))
        elif k == "r":
            self.set_state(self.rng.choice([-1.0, 1.0], N))
        elif k == "c":
            self.set_state(-np.ones(N))
        elif k == "x":
            self.mixture()

    # ---------- drawing ----------

    def redraw(self):
        self.img_s.set_data(self.s.reshape(SIDE, SIDE))
        self.img_f.set_data(aligned_field(self.W, self.theta, self.s).reshape(SIDE, SIDE))
        self.line_e.set_data(self.hist_t, self.hist_E)
        self.ax_e.relim()
        self.ax_e.autoscale_view()
        m = self.X @ self.s / N
        for bar, v in zip(self.bars, m):
            bar.set_height(v)
        self.txt.set_text(
            f"rule:     {self.rule}\n"
            f"stored:   {len(self.stored_names)}\n"
            f"dynamics: {'sync' if self.sync else 'async'}\n"
            f"next cue: {self.kind} {self.amount:.0%}\n"
            f"t = {self.t}\n"
            f"E = {self.hist_E[-1]:.2f}\n"
            f"state: {classify(m, self.stored_names)[1]}\n"
            f"{'running' if self.running else 'paused'}  {self.status}"
        )
        self.fig.canvas.draw_idle()

    def busy(self, message):
        self.txt.set_text(message)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ---------- extra windows ----------

    def show_stability(self):
        G = np.array([aligned_field(self.W, self.theta, x) for x in self.X])
        unstable = G < -1e-12
        lim = float(np.abs(G).max()) or 1.0
        cols = 6
        rows = int(np.ceil(len(G) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.3 * rows + 1.0), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, g, u, name in zip(axes.flat, G, unstable, self.stored_names):
            im = ax.imshow(g.reshape(SIDE, SIDE), cmap="RdBu", vmin=-lim, vmax=lim)
            ax.set_title(f"{name}: {int(u.sum())} unstable", fontsize=9)
        fixed = int((~unstable).all(axis=1).sum())
        fig.suptitle(
            f"rule: {self.rule}   aligned field ξ_i (h_i - θ_i) of each stored pattern, red = bit flips\n"
            f"stored patterns that are fixed points: {fixed}/{len(G)}"
        )
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
        fig.show()

    def show_spurious(self, starts=200):
        self.busy(f"running {starts}\nrandom starts…")
        counts = {}
        for _ in range(starts):
            s, _ = run(self.W, self.theta, self.rng.choice([-1.0, 1.0], N), self.rng)
            key = s.astype(np.int8).tobytes()
            counts[key] = counts.get(key, 0) + 1

        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        tally = {"stored": 0, "reversed": 0, "spurious": 0}
        shown = []
        for key, c in ranked:
            s = np.frombuffer(key, dtype=np.int8).astype(float)
            cat, text = classify(self.X @ s / N, self.stored_names)
            tally[cat] += c
            shown.append((s, f"{text} ×{c}"))
        shown = shown[:24]

        cols = 8
        rows = int(np.ceil(len(shown) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 1.7 * rows + 0.8), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (s, title) in zip(axes.flat, shown):
            ax.imshow(s.reshape(SIDE, SIDE), cmap="gray_r", vmin=-1, vmax=1)
            ax.set_title(title, fontsize=9)
        summary = ", ".join(f"{k}: {v}" for k, v in tally.items())
        fig.suptitle(f"rule: {self.rule}   {starts} random starts -> {len(counts)} distinct final states ({summary})")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.show()
        self.redraw()

    def show_capacity(self, trials=10):
        amount = self.amount
        self.busy("computing\ncapacity curves…")
        datasets = {
            "icons": np.array([self.patterns[n] for n in self.names]),
            "random": self.rng.choice([-1.0, 1.0], size=(int(0.4 * N), N)),
        }
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, (dname, data) in zip(axes, datasets.items()):
            ps = np.unique(np.linspace(1, len(data), 12).astype(int))
            for rule, learn in RULES.items():
                quality = []
                for p in ps:
                    X = data[:p]
                    W, theta = learn(X)
                    tested = self.rng.choice(p, min(p, trials), replace=False)
                    q = [X[k] @ run(W, theta, corrupt(X[k], "flip", amount, self.rng), self.rng)[0] / N for k in tested]
                    quality.append(np.mean(q))
                ax.plot(ps, quality, marker="o", label=rule)
            if dname == "random":
                ax.axvline(0.138 * N, color="0.5", ls="--", label="0.138 N")
            ax.set_title(f"{dname}, N = {N}")
            ax.set_xlabel("stored patterns P")
            ax.legend()
        axes[0].set_ylabel(f"mean overlap with cued pattern after recall\n(cue: {amount:.0%} bits flipped)")
        fig.tight_layout()
        fig.show()
        self.redraw()


def main():
    for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[key] = []
    app = App()
    plt.show()
    return app


if __name__ == "__main__":
    main()
