import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    'font.size': 12, 
    'axes.labelsize': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'cm'
})

beta = 0.99
gamma = 0.25
theta = 0.75
varphi = 3
rho_a = 0.95

def solve(tau, periods=20):
    lambda_ = (1 - theta) * (1 - beta * theta) / theta
    omega = lambda_ * ((1 + gamma * tau) / (1 - gamma) + varphi)
    psi = omega * (1 - gamma) / (1 + gamma * tau)
    b_coeff = -(1 + beta + psi)
    
    delta = (-b_coeff - np.sqrt(b_coeff**2 - 4 * beta)) / (2 * beta)
    phi = (lambda_ * (1 + varphi)) / (beta * (delta + rho_a) + b_coeff)

    a, p, y, g = np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods)
    
    a[0] = -1.0 
    p[0] = phi * a[0]
    y[0] = -((1 - gamma) / (1 + gamma * tau)) * p[0]
    g[0] = -tau * y[0]
    
    for t in range(1, periods):
        a[t] = rho_a * a[t-1]
        p[t] = delta * p[t-1] + phi * a[t]
        y[t] = -((1 - gamma) / (1 + gamma * tau)) * p[t]
        g[t] = -tau * y[t]
        
    return a, p, y, g

periods = 40
time = np.arange(periods)

tau_values = np.array([0.0, 5.0, 20.0, 40.0])
colors = sns.color_palette("tab10", n_colors=len(tau_values))

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

def add_arrows(ax, x, y, color):
    """Places standalone triangle arrowheads along the curve based on display coordinates."""
    step = 1
    idx = np.arange(0, len(x) - step, step)

    pts = ax.transData.transform(np.column_stack([x, y]))
    
    dx = pts[idx + step, 0] - pts[idx, 0]
    dy = pts[idx + step, 1] - pts[idx, 1]

    angles = np.degrees(np.arctan2(dy, dx))

    for xi, yi, ang in zip(x[idx], y[idx], angles):
        ax.scatter(
            xi, yi,
            marker=(3, 0, ang - 90),
            s=60,
            color=color,
            zorder=3
        )

a_base, _, _, _ = solve(tau=0.0, periods=periods)
axs[0, 0].plot(time, a_base, color='black', linewidth=3.5)
axs[0, 0].set_title('Productivity Shock ($a_t^i$)')

results = [] 
for idx, t_val in enumerate(tau_values):
    a, p, y, g = solve(tau=t_val, periods=periods)
    results.append((p, y, g))
    
    color = colors[idx]
    label_text = rf'$\tau = {t_val:.1f}$' if t_val > 0 else r'$\tau = 0$'
    
    axs[0, 1].plot(time, p, color=color, linewidth=3.5, label=label_text)
    axs[1, 0].plot(time, y, color=color, linewidth=3.5, label=label_text)
    axs[1, 1].plot(time, g, color=color, linewidth=3.5, label=label_text)

axs[0, 1].set_title('Domestic Price Level ($p_t^i$)')
axs[0, 1].legend(loc='best')

axs[1, 0].set_title('Output deviation ($\hat{y}_t^i$)')
axs[1, 0].legend(loc='best')

axs[1, 1].set_title('Fiscal deviation ($g_t^i$)')
axs[1, 1].legend(loc='best')

for ax in axs.flat:
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Deviation from Steady State')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, periods - 1)

plt.subplots_adjust(hspace=0.4, wspace=0.35)

fig.canvas.draw()

add_arrows(axs[0, 0], time, a_base, 'black')

for idx, (p, y, g) in enumerate(results):
    color = colors[idx]
    add_arrows(axs[0, 1], time, p, color)
    add_arrows(axs[1, 0], time, y, color)
    add_arrows(axs[1, 1], time, g, color)

plt.savefig('IRF_Plot.pdf', format='pdf', bbox_inches='tight')
plt.show()