import sys
import re
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from plot_utils import plot_instruction_distribution
from matplotlib.cm import get_cmap



INSTRUCTION_CATEGORIES = {
    'data_transfer': [
        'movb', 'movl', 'movq', 'movw', 'movabsq', 'movapd', 'movaps',
        'movsd', 'movss', 'movsbl', 'movsbw', 'movswl', 'movslq',
        'movzbl', 'movzwl', 'pushq', 'popq', 'leal', 'leaq',
        'cmovsl', 'cmovsq', 'movd', 'movdqa', 'movdqu', 'movhlps',
        'movhpd', 'movhps', 'movlhps', 'movlpd', 'movlps', 'movmskpd',
        'movsbq', 'movswq', 'movups', 'movupd', 'fld', 'fld1', 'fldl',
        'flds', 'fldz', 'fldcw', 'fldt', 'fstl', 'fsts', 'fstp',
        'fstpl', 'fstpt', 'fildl', 'fildll', 'filds', 'fistpl',
        'fistpll', 'fistps', 'cmoval', 'cmovael', 'cmovaq', 'cmovaw',
        'cmovb', 'cmovbel', 'cmovbeq', 'cmovbw', 'cmoveq', 'cmovel',
        'cmovg', 'cmovgeq', 'cmovgl', 'cmovgw', 'cmovl', 'cmovleq',
        'cmovll', 'cmovlw', 'cmovnbe', 'cmovneq', 'cmovnel', 'cmovns',
        'cmovnsl', 'cmovnpl', 'cmovnsq', 'cmovp', 'cmovs'
    ],
    'arithmetic': [
        'addl', 'addq', 'addsd', 'addss', 'addw', 'subl', 'subq',
        'subsd', 'subss', 'imull', 'imulq', 'divq', 'divsd', 'divss',
        'mulsd', 'mulss', 'negl', 'negq', 'idivq', 'addb', 'addpd',
        'addps', 'adcl', 'adcq', 'divb', 'divl', 'divpd', 'divps',
        'divw', 'fadd', 'faddl', 'faddp', 'fadds', 'fdiv', 'fdivl',
        'fdivp', 'fdivr', 'fdivrl', 'fdivrp', 'fdivrs', 'fdivs',
        'fmul', 'fmull', 'fmulp', 'fmuls', 'fsub', 'fsubp', 'fsubr',
        'fsubrl', 'fsubrp', 'fsubs', 'idivl', 'imulb', 'imulw',
        'mulb', 'mulpd', 'mulps', 'mulq', 'mulw', 'paddb', 'paddd',
        'paddq', 'paddw', 'pmaddwd', 'pmulhuw', 'pmulhw', 'pmullw',
        'pmuludq', 'psadbw', 'psubb', 'psubd', 'psubq', 'psubw',
        'psubusb', 'psubusw', 'sbbb', 'sbbl', 'sbbq', 'sbbw',
        'maxpd', 'maxps', 'maxsd', 'maxss', 'minpd', 'minps',
        'minsd', 'minss', 'pmaxub', 'pmaxsw', 'pminub', 'pminsw'
    ],
    'logical': [
        'andl', 'andpd', 'andps', 'andq', 'orl', 'orq', 'notl', 'notq',
        'xorl', 'xorpd', 'xorps', 'xorq', 'andb', 'andnpd', 'andnps',
        'andw', 'notb', 'notw', 'orb', 'orw', 'xorb', 'xorw',
        'pand', 'pandn', 'por', 'pxor', 'pcmpeqb', 'pcmpeqd',
        'pcmpeqw', 'pcmpgtb', 'pcmpgtd', 'pcmpgtw'
    ],
    'shift_and_rotate': [
        'shll', 'shlq', 'shrl', 'shrq', 'sarl', 'sarq', 'pxor',
        'rolb', 'rolq', 'rolw', 'roll', 'rorb', 'rorq', 'rorw',
        'rorl', 'pshufd', 'pshufhw', 'pshuflw', 'pslld', 'psllq',
        'psllw', 'psrad', 'psrld', 'psrlq', 'psrlw', 'psrldq',
        'shufpd', 'shufps', 'unpckhpd', 'unpckhps', 'unpcklpd',
        'unpcklps', 'punpckhbw', 'punpckhdq', 'punpckhqdq',
        'punpckhwd', 'punpcklbw', 'punpckldq', 'punpcklqdq',
        'punpcklwd', 'bswapq', 'bswapl', 'bswapw', 'bswapb',
        'pinsrw', 'pextrw'
    ],
    'control_transfer': [
        'callq', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl',
        'jle', 'jmp', 'jmpq', 'jne', 'jns', 'jp', 'js', 'retq',
        'endbr64', 'hlt', 'lock', 'rep', 'ud2'
    ],
    'comparison': [
        'cmpb', 'cmpl', 'cmpq', 'cmpw', 'comisd', 'comiss', 'testb',
        'testl', 'testq', 'testw', 'seta', 'setae', 'setb', 'setbe',
        'sete', 'setg', 'setge', 'setl', 'setle', 'setne', 'setnp',
        'setns', 'setp', 'sets', 'ucomisd', 'ucomiss', 'cmpeqpd',
        'cmpeqps', 'cmpeqsd', 'cmpeqss', 'cmplepd', 'cmpleps',
        'cmplesd', 'cmpless', 'cmpltpd', 'cmpltps', 'cmpltsd',
        'cmpltss', 'cmpneqpd', 'cmpneqps', 'cmpneqsd', 'cmpneqss',
        'cmpnlepd', 'cmpnleps', 'cmpnlesd', 'cmpnless', 'cmpnltpd',
        'cmpnltps', 'cmpnltsd', 'cmpnltss', 'cmpunordpd', 'cmpunordps',
        'fcomi', 'fcompi', 'fucomi', 'fucompi'
    ],
    'other': [
        'nop', 'nopl', 'nopw', 'endbr64', 'hlt', 'leave', 'btrq',
        'btq', 'btcq', 'bsrq', 'fabs', 'fcmovbe', 'fcmovnbe',
        'fistpl', 'fistpll', 'fistps', 'fnstcw', 'fxch', 'packuswb',
        'pause', 'cbtw', 'cltq', 'cqto', 'cvtsd2ss', 'cvtsi2sd', 'cvtsi2sdl',
        'cvtsi2ss', 'cvtsi2ssl', 'cvtss2sd', 'cvttsd2si', 'cvttss2si',
        'cwtl', 'cvtdq2pd', 'cvtps2pd', 'cvttpd2dq'
    ]
}
def categorize_instruction(instruction):
    instruction = instruction.lower()
    for category, prefixes in INSTRUCTION_CATEGORIES.items():
        for prefix in prefixes:
            if instruction.startswith(prefix):
                return category
    return 'other'

def parse_assembly_file(filename):
    instruction_list = []
    instruction_counts = defaultdict(int)
    instruction_pattern = re.compile(r'^\s*[0-9a-f]+:\s+([a-z]+[a-z0-9]*)')

    with open(filename, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith(('DWARF', 'End', 'BOLT', 'Binary')):
                match = instruction_pattern.match(line)
                if match:
                    instruction = match.group(1)
                    instruction_list.append(instruction)
                    instruction_counts[instruction] += 1
    
    return instruction_list, instruction_counts

def analyze_instructions(instructions):
    category_counts = defaultdict(int)
    
    for instr in instructions:
        category = categorize_instruction(instr)
        if category != 'other':
            category_counts[category] += 1
    
    return category_counts

def print_statistics(stats):
    print("\nBinary File Instruction Statistics")
    print("=" * 35)
    total = sum(stats.values())
    for category, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"{category:<15}: {count:>4} ({percentage:.1f}%)")

def expand_file_patterns(patterns):
    """Expand command line arguments with glob patterns."""
    filenames = []
    for pattern in patterns:
        matched_files = glob.glob(pattern)
        if not matched_files:
            print(f"Warning: No files matched pattern '{pattern}'")
        filenames.extend(matched_files)
    return filenames

def plot_instruction_distribution_stack_bar(instruction_counts, binary_name):

    categories = list(INSTRUCTION_CATEGORIES.keys())
    num_categories = len(categories)

    plt.figure(figsize=(20,12))

    cmap = get_cmap('RdYlBu')
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]

    hatches = ['o', '/', '.', '\\', '+', 'x']

    # 计算总高度用于确定标签显示阈值
    total_counts = {cat: sum(instruction_counts.get(instr, 0) for instr in instrs) 
                    for cat, instrs in INSTRUCTION_CATEGORIES.items()}
    max_height = max(total_counts.values())

    # 标签显示阈值 (柱子高度必须大于最大高度的5%才显示标签)
    label_threshold = max_height * 0.02

    for i, category in enumerate(categories):
        instructions = INSTRUCTION_CATEGORIES[category]
        counts = [(instr, instruction_counts.get(instr, 0)) for instr in instructions]
        
        counts.sort(key=lambda x: -x[1])
        top_5 = counts[:5]
        others = sum(count for _, count in counts[5:])
        
        bottom = 0
        bar_width = 0.3
        for j, (instr, count) in enumerate(top_5):
            data = np.zeros(num_categories)
            data[i] = count
            bar = plt.bar(categories, data, width=bar_width, bottom=bottom,
                        hatch=hatches[j] if j < len(hatches) else None,
                        linewidth=0.4,
                        label=f"{category}: {instr}" if i == 0 else "", 
                        color=colors[j])


            if count > label_threshold:
                for rect in bar:
                    height = rect.get_height()
                    if height > 0:
                        plt.text(rect.get_x() + rect.get_width()/2.,
                                bottom + height/2.,
                                f"{instr} {count}",
                                ha='center', va='center',
                                fontsize=12, color='white',
                                bbox=dict(facecolor='black', alpha=0.3, pad=0.5, edgecolor='none'))
            bottom += count
        
        # 添加Others部分
        if others > 0:
            data = np.zeros(num_categories)
            data[i] = others
            bar = plt.bar(categories, data, bottom=bottom,
                        width=bar_width,
                        label=f"{category}: Others" if i == 0 else "", 
                        # hatch=hatches[5] if len(hatches) > 5 else None,
                        color=colors[5])
            
            # 只在足够大的Others部分添加标签
            if others > label_threshold:
                for rect in bar:
                    height = rect.get_height()
                    if height > 0:
                        plt.text(rect.get_x() + rect.get_width()/2.,
                                bottom + height/2.,
                                f"Others {others}",
                                ha='center', va='center',
                                fontsize=12, color='white',
                                bbox=dict(facecolor='black', alpha=0.3, pad=0.5, edgecolor='none'))
            bottom += others

    plt.ylabel('Instruction Count', fontsize=18)
    plt.title(f'Instruction Distribution for {binary_name}', fontsize=14)
    plt.xticks(fontsize=18)

    output_dir = 'memcmp_stack_chart'
    filename = f"v4_{binary_name}.png"
    output_path = os.path.join(output_dir, filename)


    plt.tight_layout()
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    # plt.show()

def main():
    filenames = expand_file_patterns(sys.argv[1:])

    print(filenames)

    for filename in filenames:
        base_name = filename.split("/")[-1]
        # binary_name = base_name[:-len("_disasm_output.txt")]
        binary_name = base_name
        print(binary_name)

        instruction_list, instruction_counts = parse_assembly_file(filename)
        stats = analyze_instructions(instruction_list)
        print_statistics(stats)

        # 堆叠柱状图
        plot_instruction_distribution_stack_bar(instruction_counts, binary_name)
        
        # 柱状图
        # plot_instruction_distribution('Instruction Distribution', 'Instruction Types', stats, binary_name)

if __name__ == "__main__":
    main()