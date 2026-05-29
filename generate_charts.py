import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Set a dark theme for a premium corporate look
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", "grid.color": "#333333", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

file_path = 'Z13-Lemonade-Benchmark-96G.md'

models = []
old_tps = []
new_tps = []

# Parse the markdown table
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    in_table = False
    for line in lines:
        if '| Rank | Model | Old tok/s |' in line:
            in_table = True
            continue
        if in_table and line.startswith('|---'):
            continue
        if in_table and line.startswith('| '):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 4:
                try:
                    rank = parts[1]
                    model = parts[2].replace('**', '')
                    old = float(parts[3])
                    new = float(parts[4].replace('**', ''))
                    models.append(model)
                    old_tps.append(old)
                    new_tps.append(new)
                except ValueError:
                    continue
        if in_table and line.strip() == '':
            break

# Get top 10 models for the chart
models = models[:10]
old_tps = old_tps[:10]
new_tps = new_tps[:10]

import numpy as np
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Lemonade Benchmark: 512MB UMA vs 96GB Dedicated VRAM', fontsize=18, fontweight='bold', color='#00d2ff')

rects1 = ax.bar(x - width/2, old_tps, width, label='512MB UMA (Old)', color='#ff5252')
rects2 = ax.bar(x + width/2, new_tps, width, label='96GB VRAM (New)', color='#00e676')

ax.set_ylabel('Tokens / Second (Generation)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
os.makedirs('assets', exist_ok=True)
plt.savefig('assets/lemonade_dashboard.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Lemonade Dashboard generated successfully.")
