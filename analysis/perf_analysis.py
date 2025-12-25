import sys
import re
import pandas as pd

def parse_log(file_handle):
    data = []
    # Regex to capture the table rows
    # Expected format: "   4 | NAIVE      |          29 | ..."
    row_pattern = re.compile(r'\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)')

    for line in file_handle:
        match = row_pattern.match(line)
        if match:
            size, method, l1d, llc, instr, cycles = match.groups()
            data.append({
                "SIZE": int(size),
                "METHOD": method,
                "L1D_MISS": int(l1d),
                "LLC_MISS": int(llc),
                "INSTR": int(instr),
                "CYCLES": int(cycles)
            })
    return pd.DataFrame(data)

def main():
    # Read from Stdin (pipe) or file
    df = parse_log(sys.stdin)

    if df.empty:
        print("No data found.")
        return

    # Calculate aggregations
    # We want Mean for everything, but also Min for L1D to spot the 'clean' runs
    summary = df.groupby(['SIZE', 'METHOD']).agg({
        'L1D_MISS': ['mean', 'min', 'std'],
        'LLC_MISS': 'mean',
        'CYCLES': 'mean',
        'INSTR': 'mean'
    }).round(1)

    # Calculate IPC (Instructions Per Cycle) based on the means
    summary['IPC'] = (summary[('INSTR', 'mean')] / summary[('CYCLES', 'mean')]).round(2)

    # Print nicely
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print(summary)

if __name__ == "__main__":
    main()
