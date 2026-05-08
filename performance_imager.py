import matplotlib.pyplot as plt
import numpy as np

# Your measured data from the lab report
sizes = [64, 128, 256, 512, 1024]

data = {
    'i-j-k (naive)': [None, None, 2.73, None, 0.61],
    'i-k-j':         [None, None, 15.66, None, 16.59],
    'j-k-i':         [None, None, 0.71, None, 0.23],
    'k-i-j':         [None, None, 7.34, None, 9.70],
    'Tiled (best)':  [None, None, None, None, 18.16],
}

plt.figure(figsize=(10, 6))

for label, values in data.items():
    x = [sizes[i] for i, v in enumerate(values) if v is not None]
    y = [v for v in values if v is not None]
    plt.plot(x, y, marker='o', label=label)

plt.xscale('log', base=2)
plt.xlabel('Matrix Size N')
plt.ylabel('Performance (GFLOP/s)')
plt.title('Matrix Multiplication Performance – CPU')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

import os
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/performance.png', dpi=150)
plt.show()
print("Saved to figures/performance.png")