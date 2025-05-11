import re
from collections import defaultdict
import sys
import glob
from plot_utils import plot_instruction_distribution

def count_instructions_from_file(filename):
    instruction_pattern = re.compile(r'^\s*[0-9a-f]+:\s+([a-z]+[a-z0-9]*)')
    instruction_counts = defaultdict(int)
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                match = instruction_pattern.search(line)
                if match:
                    instruction = match.group(1)
        return instruction_counts
    except Exception as e:
        return None

def analyze_files(filenames):
    all_counts = defaultdict(int)
    per_file_counts = {}
    file_success = 0
    
    for filename in filenames:
        counts = count_instructions_from_file(filename)
        if counts is not None:
            per_file_counts[filename] = counts
            for inst, count in counts.items():
                all_counts[inst] += count
            file_success += 1

            base_name = filename.split("/")[-1]
            binary_name = base_name[:-len("_disasm_output.txt")]
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
            top_k_counts = dict(sorted_counts)
            plot_instruction_distribution('Instruction Details', 'Instruction Name', top_k_counts, binary_name, 'pre_chart')
    
    return all_counts, per_file_counts, file_success

def print_results(all_counts, per_file_counts, total_files):
    for inst, count in sorted(all_counts.items(), key=lambda x: x[1]):
        print(f"{inst}: {count}\n")
    print(f"Total unique instructions: {len(all_counts)}\n")
    print(f"Total instructions: {sum(all_counts.values())}\n")
    
    for filename, counts in per_file_counts.items():
        print(f"File: {filename}\n")
        print(f"Unique instructions: {len(counts)}\n")
        print(f"Total instructions: {sum(counts.values())}\n")
    
def expand_file_patterns(patterns):
    filenames = []
    for pattern in patterns:
        matched_files = glob.glob(pattern)
        filenames.extend(matched_files)
    return filenames

def main():
    filenames = expand_file_patterns(sys.argv[1:])
    all_counts, per_file_counts, files_processed = analyze_files(filenames)
    print_results(all_counts, per_file_counts, len(filenames))

if __name__ == "__main__":
    main()