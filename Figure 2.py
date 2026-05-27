import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

beta = 0.99
gamma = 0.25
theta = 0.75
varphi = 3
rho_a = 0.95

def solve(tau, periods=100):
    lambda_ = (1 - theta) * (1 - beta * theta) / theta
    omega = lambda_ * ((1 + gamma * tau) / (1 - gamma) + varphi)
    psi = omega * (1 - gamma) / (1 + gamma * tau)
    b_coeff = -(1 + beta + psi)

    delta = (-b_coeff - np.sqrt(b_coeff**2 - 4 * beta)) / (2 * beta)
    phi = (lambda_ * (1 + varphi)) / (beta * (delta + rho_a) + b_coeff)

    a = np.zeros(periods)
    p = np.zeros(periods)
    pi = np.zeros(periods)
    y = np.zeros(periods)
    g = np.zeros(periods)

    a[0] = -1.0
    p[0] = phi * a[0]
    pi[0] = p[0]
    y[0] = -((1 - gamma) / (1 + gamma * tau)) * p[0]
    g[0] = -tau * y[0]

    for t in range(1, periods):
        a[t] = rho_a * a[t - 1]
        p[t] = delta * p[t - 1] + phi * a[t]
        pi[t] = p[t] - p[t - 1]
        y[t] = -((1 - gamma) / (1 + gamma * tau)) * p[t]
        g[t] = -tau * y[t]

    return y, pi, g

tau_grid = np.linspace(0, 1000, 100)

std_y = []
std_pi = []
std_g = []

for t_val in tau_grid:
    y_sim, pi_sim, g_sim = solve(tau=t_val)

    std_y.append(np.std(y_sim))
    std_pi.append(np.std(pi_sim))
    std_g.append(np.std(g_sim))

colors = sns.color_palette("tab10", n_colors=4)
color = colors[3]


fig, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].plot(std_y, std_pi, color=color, linewidth=2.5)

axs[0].set_title('Stabilization Inflation Trade-Off', fontsize=16, pad=15)

axs[0].set_xlabel(
    r'Output Volatility ($\sigma_y$)',
    fontsize=14
)

axs[0].set_ylabel(
    r'Inflation Volatility ($\sigma_\pi$)',
    fontsize=14
)

axs[0].set_xlim(left=-0.005)
axs[0].set_ylim(bottom=0.05)

axs[0].xaxis.set_major_formatter(
    mtick.PercentFormatter(xmax=100, decimals=2)
)

axs[0].yaxis.set_major_formatter(
    mtick.PercentFormatter(xmax=100, decimals=3)
)

axs[1].plot(std_y, std_g, color=color, linewidth=2.5)

axs[1].set_title('Stabilization Fiscal Trade-Off', fontsize=16, pad=15)

axs[1].set_xlabel(
    r'Output Volatility ($\sigma_y$)',
    fontsize=14
)

axs[1].set_ylabel(
    r'Fiscal Volatility ($\sigma_g$)',
    fontsize=14
)

axs[1].set_xlim(left=-0.005)
axs[1].set_ylim(bottom=-0.02)

axs[1].xaxis.set_major_formatter(
    mtick.PercentFormatter(xmax=100, decimals=2)
)

axs[1].yaxis.set_major_formatter(
    mtick.PercentFormatter(xmax=100, decimals=2)
)

for ax in axs.flat:
    ax.grid(True, linestyle='--', alpha=0.7)

plt.subplots_adjust(wspace=0.35)

plt.savefig(
    'Volatility_Tradeoffs.pdf',
    format='pdf',
    bbox_inches='tight'
)

coef_pi = np.polyfit(std_y, std_pi, 1)
coef_g  = np.polyfit(std_y, std_g, 1)

m_pi, b_pi = coef_pi
m_g,  b_g  = coef_g

print("Domestic Inflation Trade-Off Line:")
print(f"sigma_pi = {m_pi:.6f} * sigma_y + {b_pi:.6f}")

print("\nFiscal Trade-Off Line:")
print(f"sigma_g = {m_g:.6f} * sigma_y + {b_g:.6f}")