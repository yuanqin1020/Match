import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

# 数据
categories = ['WN18', 'FB15k', 'OpenImages']
values1 = [395, 1031, 17666]
values2 = [473, 1829, 25150]
values3 = [525, 2145, 30833]

# 设置柱状图的宽度
bar_width = 0.23

# 设置柱状图的位置
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width
bar_positions3 = bar_positions2 + bar_width

# 绘制柱状图
plt.bar(bar_positions1, values1, width=bar_width, label='CrossETR', color='moccasin', hatch='//', edgecolor='black')
plt.bar(bar_positions2, values2, width=bar_width, label='CrossETR-Sage', color='lightblue', hatch='\\', edgecolor='black')
plt.bar(bar_positions3, values3, width=bar_width, label='CrossETR-GAT', color='lightcyan', hatch='/', edgecolor='black')

# 设置标题和轴标签
plt.xlabel('Dataset', fontsize=22)
plt.ylabel('Train Time (s)', fontsize=22)
plt.ylim(100, 110000)
plt.yscale('log')
# 设置x轴刻度标签
plt.xticks(bar_positions1 + bar_width/2, categories, fontsize=22)
plt.yticks(fontsize=22)
plt.subplots_adjust(bottom=0.6, hspace=0.3)


# 添加图例
plt.legend(fontsize=18)
plt.tight_layout()
# 保存为eps文件
plt.savefig('/Users/yuanqin/Desktop/Paper/Semantic-driven EM/submission in revision/figures/exp_train.eps', format='eps')

# 显示图形
plt.show()
# 重置matplotlib设置
plt.clf()
plt.close('all')






# 数据
categories = ['WN18', 'FB15k', 'OpenImages']
values1 = [1.3, 13, 333]
values2 = [2, 16, 351]
values3 = [4, 19, 526]

# 设置柱状图的宽度
bar_width = 0.23

# 设置柱状图的位置
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width
bar_positions3 = bar_positions2 + bar_width

# 绘制柱状图
plt.bar(bar_positions1, values1, width=bar_width, label='CrossETR', color='moccasin', hatch='//', edgecolor='black')
plt.bar(bar_positions2, values2, width=bar_width, label='CrossETR-Sage', color='lightblue', hatch='\\', edgecolor='black')
plt.bar(bar_positions3, values3, width=bar_width, label='CrossETR-GAT', color='lightcyan', hatch='/', edgecolor='black')
plt.ylim(1, 11000)
plt.yscale('log')
# 设置标题和轴标签
plt.xlabel('Dataset', fontsize=22)
plt.ylabel('Test Time (s)', fontsize=22)

# 设置x轴刻度标签
plt.xticks(bar_positions1 + bar_width/2, categories, fontsize=22)
# 设置y轴最大值
plt.yticks(fontsize=22)
plt.subplots_adjust(bottom=0.6, hspace=0.3)


# 添加图例
plt.legend(fontsize=18)
plt.tight_layout()
# 保存为eps文件
plt.savefig('/Users/yuanqin/Desktop/Paper/Semantic-driven EM/submission in revision/figures/exp_test.eps', format='eps')

# 显示图形
plt.show()
# 重置matplotlib设置
plt.clf()
plt.close('all')




categories = ['1M', '2M', '3M', '4M', '5M']
values1 = [0.58, 0.62, 0.63, 0.59, 0.61]

plt.plot(categories, values1, marker='^', markersize=10, color='blue')
plt.xlabel('Varying data size', fontsize=22)
plt.ylabel('MRR', fontsize=22)
plt.ylim(0.4, 0.7)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.tight_layout()
# 保存为eps文件
plt.savefig('/Users/yuanqin/Desktop/Paper/Semantic-driven EM/submission in revision/figures/exp_mrr.eps', format='eps')

# 显示图形
plt.show()
# 重置matplotlib设置
plt.clf()
plt.close('all')




categories = ['1M', '2M', '3M', '4M', '5M']
values1 = [310, 632, 1060, 1264, 1579]
values2 = [34.6, 70.4, 106, 141, 193]

plt.plot(categories, values1, marker='^', markersize=10, color='red', label='Train')
plt.plot(categories, values2, marker='o', markersize=10, color='blue', label='Test')
plt.xlabel('Varying data size', fontsize=22)
plt.ylabel('Time (s)', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylim(10, 10000)
plt.yscale('log')
plt.legend(fontsize=20)
plt.tight_layout()
# 保存为eps文件
plt.savefig('/Users/yuanqin/Desktop/Paper/Semantic-driven EM/submission in revision/figures/exp_time.eps', format='eps')

# 显示图形
plt.show()
# 重置matplotlib设置
plt.clf()
plt.close('all')