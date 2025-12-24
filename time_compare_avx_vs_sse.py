import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def load_and_preprocess(file_path, label):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Extract Method and Size from the 'name' column
    # Pattern handles formats like "Simd/2048"
    pattern = r'([a-zA-Z0-9_]+)/(\d+)'
    extracted = df['name'].str.extract(pattern)
    df['Method'] = extracted[0]
    df['Size'] = extracted[1].astype(float) # float handles scientific notation better
    
    # Add the Variant tag (SSE vs AVX)
    df['Variant'] = label
    
    # Create a combined Label for the legend
    df['Implementation'] = df['Method'] + " (" + df['Variant'] + ")"
    
    return df.dropna(subset=['Size'])

# Usage check
if len(sys.argv) < 3:
    print("Usage: python plot_compare.py <sse_csv> <avx_csv>")
    sys.exit(1)

sse_file = sys.argv[1]
avx_file = sys.argv[2]

# Load and Combine
df_sse = load_and_preprocess(sse_file, "SSE2")
df_avx = load_and_preprocess(avx_file, "AVX2")

if df_sse is None or df_avx is None:
    sys.exit(1)

df_total = pd.concat([df_sse, df_avx], ignore_index=True)

# Plotting
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

y_axis = 'GOps'

# We use 'hue' for Method colors and 'style' for SSE vs AVX line styles
sns.lineplot(
    data=df_total,
    x='Size',
    y=y_axis,
    hue='Method',
    style='Variant', 
    markers=True,
    markersize=8,
    linewidth=2.5,
    palette="tab10"
)

# Configuration
plt.title(f'Performance Comparison: SSE2 vs AVX2 ({y_axis})', fontsize=16, weight='bold')
plt.xlabel('Matrix Dimension (N)', fontsize=12)
plt.ylabel(f'{y_axis} (Billions of Ops/sec)', fontsize=12)

# Matrix multiplication scaling is best viewed on log-log or log-x scales
plt.xscale('log', base=2)
# plt.yscale('log') # Uncomment if the gap between Naive and Simd is too large

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(title="Implementations", bbox_to_anchor=(1.05, 1), loc='upper left')

output_file = f"comparison_{y_axis}.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Comparison graph saved to {output_file}")
