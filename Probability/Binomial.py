import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import binom

# Parameters
n = 100
x = np.arange(0, n + 1)

# Initial p
p0 = 0.5
pmf0 = binom.pmf(x, n, p0)
cdf0 = binom.cdf(x, n, p0)

# Set up figure with two subplots
fig, (ax_pmf, ax_cdf) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(bottom=0.25, hspace=0.35)

# --- PMF plot ---
bars = ax_pmf.bar(x, pmf0, color="royalblue")
ax_pmf.set_xlim(0, n)
ax_pmf.set_ylim(0, max(pmf0) * 1.2)
ax_pmf.set_xlabel(r"$k$")
ax_pmf.set_ylabel(r"$P(X = k)$")
ax_pmf.set_title(rf"PMF: $X \sim B(n={n}, p={p0:.2f})$")

# --- CDF plot ---
cdf_line, = ax_cdf.plot(x, cdf0, color="darkorange", linewidth=2)
ax_cdf.set_xlim(0, n)
ax_cdf.set_ylim(0, 1)
ax_cdf.set_xlabel("$k$")
ax_cdf.set_ylabel(r"$P(X \leq k)$")
ax_cdf.set_title(r"$CDF$")

# Slider axis
slider_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
p_slider = Slider(
    ax=slider_ax,
    label=r"$p$",
    valmin=0.0,
    valmax=1.0,
    valinit=p0,
    valstep=0.001,
)

# Update function
def update(p):
    pmf = binom.pmf(x, n, p)
    cdf = binom.cdf(x, n, p)

    # Update PMF bars
    for bar, height in zip(bars, pmf):
        bar.set_height(height)

    ax_pmf.set_title(rf"PMF: $X \sim B(n={n}, p={p:.3f})$")

    # Update CDF line
    cdf_line.set_ydata(cdf)

    fig.canvas.draw_idle()

p_slider.on_changed(update)

plt.show()