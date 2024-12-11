# 输入文件路径
input_file = 'data\\101\\char_frequencies.txt'

# 输出文件路径
output_file = 'char_freqiencies.txt'

# 打开输入文件并处理
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 去掉数字和冒号，仅保留文字部分
        text_only = ''.join([char for char in line if not char.isdigit() and char != ':']).strip()
        # 写入到新的文件
        outfile.write(text_only + '\n')

print(f"处理完成！新的文件已保存到 {output_file}")
