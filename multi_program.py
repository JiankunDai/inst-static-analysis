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

def print_statistics(stats, binary_name):
    total = sum(stats.values())
    for category, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"{category:<15}: {count:>4} ({percentage:.1f}%)")

def expand_file_patterns(patterns):
    filenames = []
    for pattern in patterns:
        matched_files = glob.glob(pattern)
        filenames.extend(matched_files)
    return filenames

def plot_instruction_distribution_stack_bar(programs_data):

    categories = list(INSTRUCTION_CATEGORIES.keys())
    num_categories = len(categories)
    num_programs = len(programs_data)
    
    # 设置图形大小和样式
    plt.figure(figsize=(20, 12))
    # plt.style.use('seaborn')
    
    # 颜色和图案设置
    cmap = plt.get_cmap('RdYlBu')
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]
    
    hatches = ['', '/', '.', '\\', '+', 'x', 'o', '-']
    
    # 计算每个类别的总指令数，用于确定标签显示阈值
    max_height = 0
    for program_name, instruction_counts in programs_data.items():
        total_counts = {cat: sum(instruction_counts.get(instr, 0) for instr in instrs) 
                       for cat, instrs in INSTRUCTION_CATEGORIES.items()}
        current_max = max(total_counts.values())
        if current_max > max_height:
            max_height = current_max
    
    label_threshold = max_height * 0.02
    
    # 设置柱状图位置和宽度
    bar_width = 0.8 / num_programs  # 动态调整宽度
    x = np.arange(num_categories)  # 类别位置
    
    # 为每个程序绘制柱状图
    for program_idx, (program_name, instruction_counts) in enumerate(programs_data.items()):
        # 计算每个类别的指令分布
        for i, category in enumerate(categories):
            instructions = INSTRUCTION_CATEGORIES[category]
            counts = [(instr, instruction_counts.get(instr, 0)) for instr in instructions]
            counts.sort(key=lambda x: -x[1])
            top_5 = counts[:5]
            others = sum(count for _, count in counts[5:])
            
            bottom = 0
            # 绘制前5种指令
            for j, (instr, count) in enumerate(top_5):
                if count > 0:
                    plt.bar(x[i] + program_idx * bar_width, count, width=bar_width,
                           bottom=bottom,
                           color=colors[j],
                           edgecolor='white', linewidth=0.5,
                           hatch=hatches[j % len(hatches)],
                           label=f"{program_name}: {instr}" if i == 0 else "")
                    bottom += count
            
            # 绘制Others部分
            if others > 0:
                plt.bar(x[i] + program_idx * bar_width, others, width=bar_width,
                       bottom=bottom,
                       color=colors[5],
                       edgecolor='white', linewidth=0.5,
                       hatch=hatches[5 % len(hatches)],
                       label=f"{program_name}: Others" if i == 0 else "")
                
    # 设置图表标题和标签
    plt.ylabel('Instruction Count', fontsize=14)
    plt.title('Instruction Distribution Comparison', fontsize=16)
    plt.xticks(x + (num_programs-1)*bar_width/2, categories, fontsize=12, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图像
    output_dir = './'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"multi_program_comparison.png")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    # plt.show()

def main():
    filenames = expand_file_patterns(sys.argv[1:])

    program_map = defaultdict(int)

    for filename in filenames:
        base_name = filename.split("/")[-1]
        binary_name = base_name

        instruction_list, instruction_counts = parse_assembly_file(filename)
        stats = analyze_instructions(instruction_list)
        print_statistics(stats, binary_name)
        program_map[binary_name] = instruction_counts
    

    # 堆叠柱状图
    plot_instruction_distribution_stack_bar(program_map)
        

if __name__ == "__main__":
    main()