"""
Hopfield network on 16x16 pixel-art icons (N = 256 neurons, states s_i in {-1, +1}).

Dynamics: s_i <- sgn(h_i - theta_i), with local field h_i = sum_j w_ij s_j and threshold theta_i.
Energy:   E = -1/2 sum_ij w_ij s_i s_j + sum_i theta_i s_i
          (never increases under asynchronous updates when W is symmetric with zero diagonal).

Modes:
  gallery    stored icons with their mean activity a, pairwise overlaps and the weight matrix
  stability  aligned field xi_i (h_i - theta_i) of every stored pattern; negative = bit the pattern cannot hold
  recall     animated recall from a corrupted cue (asynchronous, or synchronous with --sync)
  spurious   final states reached from random initial states
  capacity   recall quality vs. number of stored patterns, icons vs. random patterns, all rules

Learning rules (X: P x N matrix of stored patterns):
  hebb       W = X^T X / N
  cov        W = (X - a)^T (X - a) / N, theta = -a (1 - a^2), a = mean activity of the patterns
  storkey    Storkey (1997): incremental and local, subtracts part of the crosstalk
  pinv       W = X^T Q^-1 X / N, Q = X X^T / N (pattern overlaps); identical to hebb for orthogonal patterns

Suggested sequence:
  python hopfield_icons_v2.py gallery                     overlaps between icons and their biased activity
  python hopfield_icons_v2.py stability                   bits that Hebbian storage cannot hold
  python hopfield_icons_v2.py recall --icons heart key    Hebbian recall with few patterns
  python hopfield_icons_v2.py recall                      Hebbian recall with all icons
  python hopfield_icons_v2.py spurious                    where the network goes instead
  python hopfield_icons_v2.py stability --rule cov        bias removed
  python hopfield_icons_v2.py stability --rule pinv       all overlaps removed
  python hopfield_icons_v2.py recall --rule pinv --corrupt mask --amount 0.5
  python hopfield_icons_v2.py recall --sync --icons heart key fish bolt
                                                          synchronous updates: E may rise, 2-cycles possible
  python hopfield_icons_v2.py capacity
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

SIDE = 16
N = SIDE * SIDE

ICONS = {
    "heart": [
        "................",
        "................",
        "..####....####..",
        ".######..######.",
        "################",
        "################",
        "################",
        ".##############.",
        "..############..",
        "...##########...",
        "....########....",
        ".....######.....",
        "......####......",
        ".......##.......",
        "................",
        "................",
    ],
    "house": [
        ".......##.......",
        "......####......",
        ".....######.....",
        "....########....",
        "...##########...",
        "..############..",
        ".##############.",
        "################",
        "..############..",
        "..#..........#..",
        "..#.##....##.#..",
        "..#.##....##.#..",
        "..#....##....#..",
        "..#....##....#..",
        "..#....##....#..",
        "..############..",
    ],
    "tree": [
        ".......##.......",
        "......####......",
        ".....######.....",
        "....########....",
        "......####......",
        ".....######.....",
        "....########....",
        "...##########...",
        ".....######.....",
        "....########....",
        "...##########...",
        "..############..",
        ".##############.",
        ".......##.......",
        ".......##.......",
        ".....######.....",
    ],
    "fish": [
        "................",
        "................",
        "................",
        "......#####.....",
        "....#########...",
        "#..###########..",
        "##.#########.##.",
        "###############.",
        "###############.",
        "##.############.",
        "#..###########..",
        "....#########...",
        "......#####.....",
        "................",
        "................",
        "................",
    ],
    "key": [
        "................",
        "................",
        "................",
        "................",
        "..####..........",
        ".##..##.........",
        "##....##........",
        "##....##########",
        "##....##########",
        ".##..##...##.##.",
        "..####....##.##.",
        "................",
        "................",
        "................",
        "................",
        "................",
    ],
    "moon": [
        "................",
        ".......#####....",
        ".....######.....",
        "....#####.......",
        "...#####........",
        "...####.........",
        "..####..........",
        "..####..........",
        "..####..........",
        "..####..........",
        "...####.........",
        "...#####........",
        "....#####.......",
        ".....######.....",
        ".......#####....",
        "................",
    ],
    "umbrella": [
        "................",
        ".......##.......",
        "....########....",
        "..############..",
        ".##############.",
        "################",
        "################",
        "##.##.####.##.##",
        ".......##.......",
        ".......##.......",
        ".......##.......",
        ".......##.......",
        "...##..##.......",
        "...##..##.......",
        "...######.......",
        "................",
    ],
    "anchor": [
        ".......##.......",
        "......#..#......",
        "......#..#......",
        ".......##.......",
        "....########....",
        ".......##.......",
        ".......##.......",
        ".......##.......",
        ".......##.......",
        ".......##.......",
        ".##....##....##.",
        ".###...##...###.",
        "..###..##..###..",
        "...##########...",
        ".....######.....",
        "................",
    ],
    "cup": [
        "................",
        "....#..#..#.....",
        ".....#..#..#....",
        "....#..#..#.....",
        "................",
        "..##########....",
        "..############..",
        "..##########..#.",
        "..##########..#.",
        "..##########..#.",
        "..############..",
        "..##########....",
        "...########.....",
        ".##############.",
        "................",
        "................",
    ],
    "star": [
        ".......##.......",
        ".......##.......",
        "......####......",
        "......####......",
        "################",
        ".##############.",
        "...##########...",
        "....########....",
        "....########....",
        "...##########...",
        "...####..####...",
        "..####....####..",
        "..###......###..",
        ".###........###.",
        ".##..........##.",
        "................",
    ],
    "bolt": [
        ".........#####..",
        "........#####...",
        ".......#####....",
        "......#####.....",
        ".....#####......",
        "....##########..",
        "...##########...",
        "........#####...",
        ".......#####....",
        "......#####.....",
        ".....####.......",
        "....####........",
        "...###..........",
        "..###...........",
        "..##............",
        "................",
    ],
    "bell": [
        ".......##.......",
        "......####......",
        ".....######.....",
        "....########....",
        "....########....",
        "...##########...",
        "...##########...",
        "...##########...",
        "..############..",
        "..############..",
        ".##############.",
        "################",
        "................",
        "......####......",
        ".......##.......",
        "................",
    ],
}


def to_pattern(rows):
    rows = [r.ljust(SIDE, ".")[:SIDE] for r in rows]
    rows = (rows + ["." * SIDE] * SIDE)[:SIDE]
    return np.array([1.0 if c == "#" else -1.0 for r in rows for c in r])


def hebb(X):
    """w_ij = (1/N) sum_mu xi_i^mu xi_j^mu, w_ii = 0.
    Field of stored pattern nu: h_i = xi_i^nu (signal) + sum_{mu != nu} xi_i^mu m_mu,nu (crosstalk)."""
    W = X.T @ X / X.shape[1]
    np.fill_diagonal(W, 0.0)
    return W, np.zeros(X.shape[1])


def cov(X):
    """Hebb on patterns centered by their mean activity a, plus a uniform threshold.
    Field of a stored pattern with mean close to a: h_i = (xi_i - a)(1 - a^2) + crosstalk, so
    theta = -a (1 - a^2) gives xi_i = +1 and xi_i = -1 the same margin, 1 - a^2."""
    a = X.mean()
    Y = X - a
    W = Y.T @ Y / X.shape[1]
    np.fill_diagonal(W, 0.0)
    return W, np.full(X.shape[1], -a * (1 - a * a))


def storkey(X):
    n = X.shape[1]
    W = np.zeros((n, n))
    for x in X:
        h = W @ x
        H = h[:, None] - W * x[None, :]  # H[i, j] = sum over k != i, j of w_ik x_k
        W += (np.outer(x, x) - x[:, None] * H.T - H * x[None, :]) / n
        np.fill_diagonal(W, 0.0)
    return W, np.zeros(n)


def pinv(X):
    W = np.linalg.pinv(X) @ X
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    return W, np.zeros(X.shape[1])


RULES = {"hebb": hebb, "cov": cov, "storkey": storkey, "pinv": pinv}


def energy(W, theta, s):
    return -0.5 * s @ W @ s + theta @ s


def aligned_field(W, theta, s):
    """s_i (h_i - theta_i): negative where neuron i flips if updated now."""
    return s * (W @ s - theta)


def run(W, theta, s0, rng, sync=False, max_sweeps=50, record_every=0):
    """Asynchronous: one neuron at a time in random order, until a full sweep changes nothing.
    Synchronous: all neurons at once, until a fixed point or a 2-cycle."""
    s = s0.copy()
    trace = [(0, s.copy())]
    t = 0
    if sync:
        for _ in range(max_sweeps):
            h = W @ s - theta
            s = np.where(np.abs(h) > 1e-12, np.sign(h), s)
            t += s.size
            trace.append((t, s.copy()))
            if any(np.array_equal(s, past) for _, past in trace[-3:-1]):
                break
        return s, trace
    for _ in range(max_sweeps):
        changed = False
        for i in rng.permutation(s.size):
            h = W[i] @ s - theta[i]
            if abs(h) > 1e-12 and np.sign(h) != s[i]:
                s[i] = -s[i]
                changed = True
            t += 1
            if record_every and t % record_every == 0:
                trace.append((t, s.copy()))
        if not changed:
            break
    if trace[-1][0] != t:
        trace.append((t, s.copy()))
    return s, trace


def corrupt(x, kind, amount, rng):
    y = x.copy()
    if kind == "flip":
        idx = rng.choice(y.size, int(round(amount * y.size)), replace=False)
        y[idx] = -y[idx]
    else:
        rows = int(round(amount * SIDE))
        y.reshape(SIDE, SIDE)[SIDE - rows :] = -1.0
    return y


def classify(m, names, tol=0.9):
    """Nearest stored pattern by |overlap|; below tol the state is called spurious."""
    k = int(np.argmax(np.abs(m)))
    if abs(m[k]) < tol:
        return "spurious", "spurious"
    if m[k] > 0:
        return "stored", names[k]
    return "reversed", f"-{names[k]}"


def show(ax, s, title):
    img = ax.imshow(s.reshape(SIDE, SIDE), cmap="gray_r", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return img


def gallery(X, names, W, theta, rule):
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(rows, cols * 3)
    for k, (x, name) in enumerate(zip(X, names)):
        ax = show(fig.add_subplot(gs[k // cols, k % cols]), x, "").axes
        ax.set_title(f"{name}  a = {x.mean():+.2f}", fontsize=8)

    M = X @ X.T / N
    ax_o = fig.add_subplot(gs[:, cols : 2 * cols])
    im = ax_o.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax_o.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6)
    ax_o.set_xticks(range(len(names)))
    ax_o.set_xticklabels(names, rotation=90, fontsize=7)
    ax_o.set_yticks(range(len(names)))
    ax_o.set_yticklabels(names, fontsize=7)
    ax_o.set_title("pattern overlaps m_mu,nu", fontsize=9)
    fig.colorbar(im, ax=ax_o, fraction=0.046)

    ax_w = fig.add_subplot(gs[:, 2 * cols :])
    lim = np.abs(W).max() or 1.0
    im = ax_w.imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim)
    title = f"weights W ({rule})"
    if np.any(theta):
        title += f",  θ = {theta.mean():+.2f}"
    ax_w.set_title(title, fontsize=9)
    ax_w.set_xticks([])
    ax_w.set_yticks([])
    fig.colorbar(im, ax=ax_w, fraction=0.046)
    fig.tight_layout()
    plt.show()


def stability(X, names, W, theta, rule):
    G = np.array([aligned_field(W, theta, x) for x in X])
    unstable = G < -1e-12
    lim = np.abs(G).max() or 1.0
    cols = 6
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.3 * rows + 1.0), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, g, u, name in zip(axes.flat, G, unstable, names):
        im = ax.imshow(g.reshape(SIDE, SIDE), cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_title(f"{name}: {int(u.sum())} unstable", fontsize=9)
    fixed = int((~unstable).all(axis=1).sum())
    fig.suptitle(
        f"rule: {rule}   aligned field ξ_i (h_i - θ_i) of each stored pattern, red = bit flips\n"
        f"stored patterns that are fixed points: {fixed}/{len(names)}"
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
    plt.show()


def recall(X, names, W, theta, rule, target, kind, amount, sync, frame_every, rng):
    k = names.index(target)
    cue = corrupt(X[k], kind, amount, rng)
    _, trace = run(W, theta, cue, rng, sync=sync, record_every=frame_every)
    steps = np.array([t for t, _ in trace])
    states = [s for _, s in trace]
    energies = np.array([energy(W, theta, s) for s in states])
    overlaps = np.array([X @ s / N for s in states])
    fields = np.array([aligned_field(W, theta, s) for s in states])
    lim = np.abs(fields).max() or 1.0

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 4)
    show(fig.add_subplot(gs[0, 0]), X[k], f"stored: {names[k]}")
    show(fig.add_subplot(gs[1, 0]), cue, f"cue: {kind} {amount:.0%}")
    ax_s = fig.add_subplot(gs[0, 1])
    img = show(ax_s, states[0], "state, t = 0")
    ax_f = fig.add_subplot(gs[1, 1])
    fimg = ax_f.imshow(fields[0].reshape(SIDE, SIDE), cmap="RdBu", vmin=-lim, vmax=lim)
    ax_f.set_title("s_i (h_i - θ_i), red = will flip", fontsize=9)
    ax_f.set_xticks([])
    ax_f.set_yticks([])

    ax_e = fig.add_subplot(gs[0, 2:])
    ax_e.plot(steps, energies, color="0.8", marker="o" if sync else None)
    (line,) = ax_e.plot([], [], color="C0", marker="o" if sync else None)
    ax_e.set_xlabel("neuron updates")
    ax_e.set_ylabel("energy E")

    ax_m = fig.add_subplot(gs[1, 2:])
    bars = ax_m.bar(range(len(names)), overlaps[0], color="C0")
    bars[k].set_color("C1")
    ax_m.axhline(0, color="0.5", lw=0.8)
    ax_m.set_ylim(-1.05, 1.05)
    ax_m.set_xticks(range(len(names)))
    ax_m.set_xticklabels(names, rotation=45, ha="right")
    ax_m.set_ylabel("overlap m")

    result = classify(overlaps[-1], names)[1]
    if sync and len(states) >= 3 and np.array_equal(states[-1], states[-3]) and not np.array_equal(states[-1], states[-2]):
        result += " (2-cycle)"
    dynamics = "synchronous" if sync else "asynchronous"
    fig.suptitle(f"rule: {rule}   {dynamics} updates   final state: {result}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(f):
        img.set_data(states[f].reshape(SIDE, SIDE))
        ax_s.set_title(f"state, t = {steps[f]}", fontsize=9)
        fimg.set_data(fields[f].reshape(SIDE, SIDE))
        line.set_data(steps[: f + 1], energies[: f + 1])
        for bar, v in zip(bars, overlaps[f]):
            bar.set_height(v)
        return [img, fimg, line, *bars]

    anim = FuncAnimation(fig, update, frames=len(states), interval=400 if sync else 50, repeat=False)
    plt.show()
    return anim


def spurious(X, names, W, theta, rule, starts, rng):
    counts = {}
    for _ in range(starts):
        s, _ = run(W, theta, rng.choice([-1.0, 1.0], N), rng)
        key = s.astype(np.int8).tobytes()
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    tally = {"stored": 0, "reversed": 0, "spurious": 0}
    shown = []
    for key, c in ranked:
        s = np.frombuffer(key, dtype=np.int8).astype(float)
        cat, text = classify(X @ s / N, names)
        tally[cat] += c
        shown.append((s, f"{text} ×{c}"))
    shown = shown[:24]

    cols = 8
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.7 * rows + 0.8), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (s, title) in zip(axes.flat, shown):
        show(ax, s, title)
    summary = ", ".join(f"{k}: {v}" for k, v in tally.items())
    fig.suptitle(f"rule: {rule}   {starts} random starts -> {len(counts)} distinct final states ({summary})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


def capacity(amount, trials, rng):
    datasets = {
        "icons": np.array([to_pattern(r) for r in ICONS.values()]),
        "random": rng.choice([-1.0, 1.0], size=(int(0.4 * N), N)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (dname, data) in zip(axes, datasets.items()):
        ps = np.unique(np.linspace(1, len(data), 12).astype(int))
        for rule, learn in RULES.items():
            quality = []
            for p in ps:
                X = data[:p]
                W, theta = learn(X)
                tested = rng.choice(p, min(p, trials), replace=False)
                q = [X[k] @ run(W, theta, corrupt(X[k], "flip", amount, rng), rng)[0] / N for k in tested]
                quality.append(np.mean(q))
            ax.plot(ps, quality, marker="o", label=rule)
        if dname == "random":
            ax.axvline(0.138 * N, color="0.5", ls="--", label="0.138 N")
        ax.set_title(f"{dname}, N = {N}")
        ax.set_xlabel("stored patterns P")
        ax.legend()
    axes[0].set_ylabel(f"mean overlap with cued pattern after recall\n(cue: {amount:.0%} bits flipped)")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mode", nargs="?", default="recall", choices=["gallery", "stability", "recall", "spurious", "capacity"]
    )
    parser.add_argument("--rule", default="hebb", choices=RULES)
    parser.add_argument("--icons", nargs="+", default=list(ICONS), choices=ICONS, help="icons to store")
    parser.add_argument("--target", choices=ICONS, help="icon to cue in recall mode (default: first stored)")
    parser.add_argument("--corrupt", default="flip", choices=["flip", "mask"], help="flip random bits or blank the bottom rows")
    parser.add_argument("--amount", type=float, help="fraction of bits flipped or rows masked (default 0.25; 0.1 in capacity mode)")
    parser.add_argument("--sync", action="store_true", help="synchronous updates in recall mode")
    parser.add_argument("--starts", type=int, default=200, help="random initial states in spurious mode")
    parser.add_argument("--trials", type=int, default=10, help="cued patterns per point in capacity mode")
    parser.add_argument("--frame-every", type=int, default=8, help="single-neuron updates between animation frames")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.mode == "capacity":
        capacity(0.1 if args.amount is None else args.amount, args.trials, rng)
        return

    names = list(dict.fromkeys(args.icons))
    X = np.array([to_pattern(ICONS[n]) for n in names])
    W, theta = RULES[args.rule](X)

    if args.mode == "gallery":
        gallery(X, names, W, theta, args.rule)
    elif args.mode == "stability":
        stability(X, names, W, theta, args.rule)
    elif args.mode == "spurious":
        spurious(X, names, W, theta, args.rule, args.starts, rng)
    else:
        target = args.target or names[0]
        if target not in names:
            parser.error(f"--target {target} is not among the stored icons")
        amount = 0.25 if args.amount is None else args.amount
        recall(X, names, W, theta, args.rule, target, args.corrupt, amount, args.sync, args.frame_every, rng)


if __name__ == "__main__":
    main()
