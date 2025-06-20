#!/usr/bin/env python3
"""
Run and visualize the optimization benchmark comparison.
"""

import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import sys

def run_rust_benchmark():
    """Run the Rust benchmark and parse output."""
    print("Building benchmark binary...")
    
    # Build the benchmark
    cmd = ["cargo", "build", "--release", "--bin", "benchmark_comparison"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        return None
    
    print("Running benchmark...")
    
    # Run the benchmark
    cmd = ["cargo", "run", "--release", "--bin", "benchmark_comparison"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Benchmark failed:")
        print(result.stderr)
        return None
    
    return result.stdout

def parse_benchmark_output(output):
    """Parse the benchmark output into structured data."""
    
    results = {
        'small': {'orig': {}, 'opt': {}},
        'medium': {'orig': {}, 'opt': {}}
    }
    
    lines = output.split('\n')
    current_system = None
    current_impl = None
    
    for i, line in enumerate(lines):
        # Detect system size
        if "System size: N=" in line:
            n = int(re.search(r'N=(\d+)', line).group(1))
            current_system = 'small' if n == 24 else 'medium'
        
        # Detect implementation
        elif "Original implementation:" in line:
            current_impl = 'orig'
        elif "Optimized implementation:" in line:
            current_impl = 'opt'
        
        # Parse metrics
        elif current_system and current_impl:
            if "Time:" in line and "seconds" in line:
                time = float(re.search(r'(\d+\.\d+) seconds', line).group(1))
                results[current_system][current_impl]['time'] = time
                
                # Check for speedup
                speedup_match = re.search(r'\((\d+\.\d+)x speedup\)', line)
                if speedup_match:
                    results[current_system][current_impl]['speedup'] = float(speedup_match.group(1))
            
            elif "MC steps/sec:" in line:
                steps_per_sec = float(re.search(r'MC steps/sec: (\d+)', line).group(1))
                results[current_system][current_impl]['steps_per_sec'] = steps_per_sec
            
            elif "Memory:" in line and "MB" in line:
                memory = float(re.search(r'Memory: (\d+\.\d+) MB', line).group(1))
                results[current_system][current_impl]['memory'] = memory
                
                # Check for reduction
                reduction_match = re.search(r'\((\d+)% reduction\)', line)
                if reduction_match:
                    results[current_system][current_impl]['mem_reduction'] = float(reduction_match.group(1))
            
            elif "<cos θ>" in line:
                cos_match = re.search(r'<cos θ> = (-?\d+\.\d+) ± (\d+\.\d+)', line)
                if cos_match:
                    results[current_system][current_impl]['cos_theta'] = float(cos_match.group(1))
                    results[current_system][current_impl]['cos_error'] = float(cos_match.group(2))
            
            elif "Susceptibility:" in line:
                chi = float(re.search(r'Susceptibility: (\d+\.\d+)', line).group(1))
                results[current_system][current_impl]['susceptibility'] = chi
            
            elif "Acceptance:" in line:
                acc = float(re.search(r'Acceptance: (\d+\.\d+)%', line).group(1))
                results[current_system][current_impl]['acceptance'] = acc
            
            elif "Correctness check:" in line:
                passed = "PASSED" in line
                results[current_system]['correctness'] = passed
    
    return results

def create_performance_plot(results):
    """Create performance comparison plots."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    systems = ['small', 'medium']
    labels = ['N=24', 'N=48']
    
    # 1. Speedup comparison
    speedups = [results[sys]['opt'].get('speedup', 1.0) for sys in systems]
    bars1 = ax1.bar(labels, speedups, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax1.set_ylabel('Speedup Factor')
    ax1.set_title('Performance Speedup')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, speedup in zip(bars1, speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom')
    
    # 2. Memory usage comparison
    mem_orig = [results[sys]['orig'].get('memory', 0) for sys in systems]
    mem_opt = [results[sys]['opt'].get('memory', 0) for sys in systems]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars2_1 = ax2.bar(x - width/2, mem_orig, width, label='Original', color='#95a5a6')
    bars2_2 = ax2.bar(x + width/2, mem_opt, width, label='Optimized', color='#27ae60')
    
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Footprint')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add reduction percentages
    for i, (orig, opt) in enumerate(zip(mem_orig, mem_opt)):
        if orig > 0:
            reduction = (orig - opt) / orig * 100
            ax2.text(i, max(orig, opt) + 0.5, f'-{reduction:.0f}%', 
                    ha='center', fontsize=9)
    
    # 3. MC steps per second
    steps_orig = [results[sys]['orig'].get('steps_per_sec', 0) for sys in systems]
    steps_opt = [results[sys]['opt'].get('steps_per_sec', 0) for sys in systems]
    
    bars3_1 = ax3.bar(x - width/2, steps_orig, width, label='Original', color='#95a5a6')
    bars3_2 = ax3.bar(x + width/2, steps_opt, width, label='Optimized', color='#27ae60')
    
    ax3.set_ylabel('MC Steps/Second')
    ax3.set_title('Throughput Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Format y-axis for thousands
    ax3.ticklabel_format(style='plain', axis='y')
    
    # 4. Correctness verification
    ax4.axis('off')
    
    # Create correctness table
    table_data = []
    headers = ['System', 'Speedup', 'Memory\nReduction', '<cos θ>\nAgreement', 'χ\nAgreement', 'Status']
    
    for sys, label in zip(systems, labels):
        speedup = results[sys]['opt'].get('speedup', 1.0)
        mem_red = results[sys]['opt'].get('mem_reduction', 0)
        
        # Check observable agreement
        cos_orig = results[sys]['orig'].get('cos_theta', 0)
        cos_opt = results[sys]['opt'].get('cos_theta', 0)
        cos_err_orig = results[sys]['orig'].get('cos_error', 0.001)
        cos_err_opt = results[sys]['opt'].get('cos_error', 0.001)
        cos_diff = abs(cos_orig - cos_opt)
        cos_tol = 3 * np.sqrt(cos_err_orig**2 + cos_err_opt**2)
        cos_ok = '✓' if cos_diff < max(cos_tol, 0.001) else '✗'
        
        chi_orig = results[sys]['orig'].get('susceptibility', 0)
        chi_opt = results[sys]['opt'].get('susceptibility', 0)
        chi_diff = abs(chi_orig - chi_opt) / chi_orig if chi_orig > 0 else 0
        chi_ok = '✓' if chi_diff < 0.05 else '✗'
        
        correctness = results[sys].get('correctness', False)
        status = '✓ PASSED' if correctness else '✗ FAILED'
        
        table_data.append([
            label,
            f'{speedup:.1f}x',
            f'{mem_red:.0f}%',
            cos_ok,
            chi_ok,
            status
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the status
    for i in range(1, len(table_data) + 1):
        if '✓ PASSED' in table_data[i-1][-1]:
            table[(i, -1)].set_facecolor('#d5f4e6')
        else:
            table[(i, -1)].set_facecolor('#f8d7da')
    
    ax4.set_title('Correctness Verification', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'optimization_benchmark_{datetime.now():%Y%m%d_%H%M%S}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {filename}")
    
    return fig

def print_summary(results):
    """Print a summary of the benchmark results."""
    
    print("\n" + "="*60)
    print("OPTIMIZATION BENCHMARK SUMMARY")
    print("="*60)
    
    for sys, label in [('small', 'N=24'), ('medium', 'N=48')]:
        print(f"\n{label}:")
        
        speedup = results[sys]['opt'].get('speedup', 1.0)
        mem_reduction = results[sys]['opt'].get('mem_reduction', 0)
        correctness = results[sys].get('correctness', False)
        
        print(f"  Performance gain: {speedup:.1f}x")
        print(f"  Memory savings: {mem_reduction:.0f}%")
        print(f"  Correctness: {'✓ PASSED' if correctness else '✗ FAILED'}")
        
        # Detailed comparison
        orig_steps = results[sys]['orig'].get('steps_per_sec', 0)
        opt_steps = results[sys]['opt'].get('steps_per_sec', 0)
        
        if orig_steps > 0:
            print(f"  Throughput: {orig_steps:.0f} → {opt_steps:.0f} steps/sec")
    
    print("\n" + "="*60)

def main():
    """Run the complete benchmark analysis."""
    
    # Run the Rust benchmark
    output = run_rust_benchmark()
    
    if output is None:
        print("Failed to run benchmark!")
        sys.exit(1)
    
    # Save raw output
    with open('benchmark_output.txt', 'w') as f:
        f.write(output)
    print("Raw output saved to: benchmark_output.txt")
    
    # Parse results
    results = parse_benchmark_output(output)
    
    # Create visualization
    fig = create_performance_plot(results)
    plt.show()
    
    # Print summary
    print_summary(results)
    
    # Save results as CSV for further analysis
    data = []
    for sys in ['small', 'medium']:
        for impl in ['orig', 'opt']:
            row = {
                'system': sys,
                'implementation': impl,
                **results[sys][impl]
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv('benchmark_results.csv', index=False)
    print("\nDetailed results saved to: benchmark_results.csv")

if __name__ == "__main__":
    main()