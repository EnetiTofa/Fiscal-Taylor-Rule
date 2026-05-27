import numpy as np
import matplotlib.pyplot as plt
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

def solve(tau, periods=100, n=1.0):
    """
    Solves the model dynamics for a given fiscal response intensity (tau)
    and returns the variances of output, inflation, and government spending.
    The parameter n specifies the member state's relative share of union GDP.
    """
    lambda_ = (1 - theta) * (1 - beta * theta) / theta
    omega = lambda_ * ((1 + gamma * tau) / (1 - gamma) + varphi)
    psi = omega * (1 - gamma) / (1 + gamma * tau)
    b_coeff = -(1 + beta + psi)
    
    delta = (-b_coeff - np.sqrt(b_coeff**2 - 4 * beta)) / (2 * beta)
    phi = (lambda_ * (1 + varphi)) / (beta * (delta + rho_a) + b_coeff)

    a, p, pi, y, g = np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods), np.zeros(periods)
    
    # Initial shocks and period 0 values
    a[0] = -1.0
    p[0] = phi * a[0]
    pi[0] = p[0]
    y[0] = -((1 - gamma) / (1 + gamma * tau)) * p[0]
    g[0] = n * -tau * y[0]
    
    for t in range(1, periods):
        a[t] = rho_a * a[t-1]
        p[t] = delta * p[t-1] + phi * a[t]
        pi[t] = p[t] - p[t-1]
        y[t] = -((1 - gamma) / (1 + gamma * tau)) * p[t]
        g[t] = n * -tau * y[t]
        
    return np.var(y), np.var(pi), np.var(g)

tau_grid = np.linspace(0.01, 50, 500)
omega_pi = 1.0

scenarios = [
    {'omega_g': 0.1, 'n': 1.0,  'title': r'Domestic ($n=1$)'},
    {'omega_g': 0.1, 'n': 0.05, 'title': r'Centralised ($n=0.05$)'}
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, scenario in zip(axes, scenarios):
    omega_g = scenario['omega_g']
    n_val = scenario['n']
    loss = []
    
    for t_val in tau_grid:
        var_y, var_pi, var_g = solve(tau=t_val, n=n_val)
        L = var_y + omega_pi * var_pi + omega_g * var_g
        loss.append(L)
    
    loss = np.array(loss)
    opt_idx = np.argmin(loss)
    tau_opt = tau_grid[opt_idx]
    loss_opt = loss[opt_idx]
    colors = sns.color_palette("tab10", n_colors=4)
    color = colors[3]

    ax.plot(tau_grid, loss, color='black', linewidth=2.5)
    
    ax.plot(tau_opt, loss_opt, marker='o', markersize=8, color=color, 
            label=rf'Optimal $\tau^* \approx {tau_opt:.2f}$, $L^* \approx {loss_opt:.4f}$')
    
    ax.set_title(scenario['title'], fontsize=16, pad=15)
    ax.set_xlabel(r'Fiscal Response Intensity ($\tau$)', fontsize=14)
    ax.set_ylabel(r'Social Loss ($L$)', fontsize=14)
    ax.legend(loc='upper center', fontsize=13, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.7)

plt.subplots_adjust(wspace=0.35)
plt.tight_layout()

plt.savefig('Loss_Function.pdf', format='pdf', bbox_inches='tight')
plt.show()