import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import hypergeom

# Initial parameters
N0 = 100     # population size
K0 = 20      # successes in population
n0 = 20      # draws

# Support of X
x = np.arange(0, n0 + 1)

# Initial PMF and CDF
pmf0 = hypergeom.pmf(x, N0, K0, n0)
cdf0 = hypergeom.cdf(x, N0, K0, n0)

# Figure with two subplots
fig, (ax_pmf, ax_cdf) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(bottom=0.30, hspace=0.35)

# --- PMF plot ---
pmf_bars = ax_pmf.bar(x, pmf0, color="seagreen")
ax_pmf.set_xlim(0, n0)
ax_pmf.set_ylim(0, max(pmf0) * 1.2)
ax_pmf.set_xlabel(r"$k$")
ax_pmf.set_ylabel(r"$P(X = k)$")
ax_pmf.set_title(
    rf"PMF: $X \sim \mathrm{{Hypergeom}}(N={N0}, K={K0}, n={n0})$"
)

# --- CDF plot ---
cdf_line, = ax_cdf.plot(x, cdf0, color="darkorange", linewidth=2)
ax_cdf.set_xlim(0, n0)
ax_cdf.set_ylim(0, 1)
ax_cdf.set_xlabel(r"$k$")
ax_cdf.set_ylabel(r"$P(X \leq k)$")
ax_cdf.set_title("CDF")

# --- Slider axes ---
N_ax = fig.add_axes([0.15, 0.20, 0.7, 0.04])
K_ax = fig.add_axes([0.15, 0.14, 0.7, 0.04])
n_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04])

N_slider = Slider(N_ax, r"$N$", valmin=10, valmax=300, valinit=N0, valstep=1)
K_slider = Slider(K_ax, r"$K$", valmin=0, valmax=N0, valinit=K0, valstep=1)
n_slider = Slider(n_ax, r"$n$", valmin=1, valmax=100, valinit=n0, valstep=1)

# --- Update function ---
def update(val):
    N = int(N_slider.val)
    K = int(K_slider.val)
    n = int(n_slider.val)

    # Ensure valid relationships
    K = min(K, N)
    n = min(n, N)

    # Update slider bounds dynamically
    K_slider.valmax = N
    n_slider.valmax = N

    # Recompute support
    x = np.arange(0, n + 1)

    # Recompute PMF and CDF
    pmf = hypergeom.pmf(x, N, K, n)
    cdf = hypergeom.cdf(x, N, K, n)

    # --- Update PMF plot ---
    ax_pmf.clear()
    ax_pmf.bar(x, pmf, color="seagreen")
    ax_pmf.set_xlim(0, n)
    ax_pmf.set_ylim(0, max(pmf) * 1.2)
    ax_pmf.set_xlabel(r"$k$")
    ax_pmf.set_ylabel(r"$P(X = k)$")
    ax_pmf.set_title(
        rf"PMF: $X \sim \mathrm{{Hypergeom}}(N={N}, K={K}, n={n})$"
    )

    # --- Update CDF plot ---
    ax_cdf.clear()
    ax_cdf.plot(x, cdf, color="darkorange", linewidth=2)
    ax_cdf.set_xlim(0, n)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel(r"$k$")
    ax_cdf.set_ylabel(r"$P(X \leq k)$")
    ax_cdf.set_title("CDF")

    fig.canvas.draw_idle()

# Connect sliders
N_slider.on_changed(update)
K_slider.on_changed(update)
n_slider.on_changed(update)

plt.show()