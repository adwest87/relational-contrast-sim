#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('thermalization_test.csv')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Order parameter vs temperature
ax1.semilogx(df['T'], df['mean_abs_cos'], 'o-', markersize=8)
ax1.set_xlabel('Temperature T')
ax1.set_ylabel('<|cos θ|>')
ax1.set_title('Order Parameter')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Susceptibility vs temperature
ax2.loglog(df['T'], df['chi'], 'o-', markersize=8)
ax2.set_xlabel('Temperature T')
ax2.set_ylabel('χ')
ax2.set_title('Susceptibility')
ax2.grid(True, alpha=0.3)

# Binder cumulant vs temperature
ax3.semilogx(df['T'], df['binder'], 'o-', markersize=8)
ax3.axhline(y=2/3, color='red', linestyle='--', label='2/3 (ordered)')
ax3.axhline(y=0.61, color='green', linestyle='--', label='0.61 (3D Ising)')
ax3.set_xlabel('Temperature T')
ax3.set_ylabel('U₄')
ax3.set_title('Binder Cumulant')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim(-0.1, 0.8)

plt.tight_layout()
plt.savefig('thermalization_test.png', dpi=150)
plt.show()

# Print summary
print("Temperature limits:")
print(f"High T (T=10): <|M|> = {df[df['T']==10]['mean_abs_cos'].values[0]:.3f}")
print(f"Low T (T=0.02): <|M|> = {df[df['T']==0.02]['mean_abs_cos'].values[0]:.3f}")

