# plot_utils.py
import matplotlib.pyplot as plt
import os

def plot_instruction_distribution(title, xlabel, stats, binary_name, output_dir="chart"):
    """
    绘制指令分布直方图并保存为PNG文件
    
    参数:
        stats: dict {category: count} 指令分类统计结果
        binary_name: str 二进制文件名(用于输出文件名)
        output_dir: str 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备文件名和路径
    filename = f"v2_{binary_name}.png"
    output_path = os.path.join(output_dir, filename)
    
    # 准备绘图数据
    labels = list(stats.keys())
    values = list(stats.values())
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color='skyblue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')
    
    # 设置图表元数据
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title + f' for {binary_name}')
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved plot to: {output_path}")