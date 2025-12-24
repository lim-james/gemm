import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re

file_path = "results.csv" if len(sys.argv) < 2 else sys.argv[1]

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

pattern = r'([a-zA-Z0-9_]+)/(\d+)'
df[['Method', 'Size']] = df['name'].str.extract(pattern)
df['Size'] = df['Size'].astype(int)

df = df.dropna(subset=['Size'])

plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

y_axis = 'GOps'

sns.lineplot(
    data=df,
    x='Size',
    y=y_axis,
    hue='Method',
    style='Method',
    markers=True,
    dashes=False,
    linewidth=2.5,
    palette="viridis"
)

plt.title(f'Matrix Multiplication Performance ({y_axis})', fontsize=16, weight='bold')
plt.xlabel('Matrix Dimension (N)', fontsize=12)
plt.ylabel(y_axis, fontsize=12)

plt.xscale('log', base=2) 
# plt.yscale('log')
#lt.xscale('linear') 
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.minorticks_on()

output_file = f"benchmark_graph_{y_axis}.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Graph saved to {output_file}")
