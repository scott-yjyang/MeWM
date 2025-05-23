import re

def replace_multiple_spaces_with_tabs(input_file, output_file):
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()
    
    with open(output_file, 'w') as f_out:
        for line in lines:
            # 使用正则表达式将连续两个及以上的空格替换为单个制表符
            modified_line = re.sub(r' {2,}', '\t', line)
            f_out.write(modified_line)


# 示例用法
# replace_multiple_spaces_with_tabs('liver/val_paired_all_hcc_under_50_new.txt', 'liver/val_paired_all_hcc_under_50_new2.txt')


def check_tab_columns(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 检查每行的Tab拆分后的列数
    column_counts = []
    for line in lines:
        columns = line.split('\t')  # 按Tab拆分
        column_counts.append(len(columns))
    
    # 判断所有行的列数是否一致
    if len(set(column_counts)) == 1:
        print(f"每行按照Tab拆分后都有固定的列数：{column_counts[0]}")
    else:
        print("每行按照Tab拆分后的列数不一致：")
        for i, count in enumerate(column_counts):
            print(f"第 {i + 1} 行有 {count} 列")

# check_tab_columns('/home/yyang303/project/TextoMorph/Diffusion/cross_eval/liver/val_paired_all_hcc_under_50_new2.txt')


def extract_second_last_column(input_file, output_file):
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()
    
    with open(output_file, 'w') as f_out:
        for line in lines:
            # 按Tab拆分每行
            columns = line.strip().split('\t')
            # 检查是否有足够的列
            if len(columns) >= 4:
                # 获取倒数第四项
                second_last_item = columns[-4]
                f_out.write(second_last_item + '\n')
            else:
                print(f"警告：以下行列数不足，跳过处理：{line.strip()}")

# 示例用法
# extract_second_last_column('/home/yyang303/project/TextoMorph/Diffusion/cross_eval/liver/val_paired_all_hcc_under_50_new.txt', '/home/yyang303/project/TextoMorph/Diffusion/ddpm/con_report/surgery_action.txt')
# import random
# from collections import defaultdict
# from pathlib import Path

# def remove_duplicates(input_file, output_file):
#     # 使用集合来存储唯一的行
#     unique_lines = set()
    
#     # 读取文件并去重
#     with open(input_file, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if line and not line.startswith('...'):
#                 unique_lines.add(line)
    
#     # 写入去重后的结果
#     with open(output_file, 'w') as f:
#         for line in sorted(unique_lines):
#             f.write(line + '\n')
    
#     return len(unique_lines)

# def main():
#     # 处理训练集
#     train_input = "liver/train_paired_all_hcc_under_50.txt"
#     train_output = "liver/train_paired_all_hcc_under_50_unique.txt"
#     train_unique_count = remove_duplicates(train_input, train_output)
    
#     # 处理验证集
#     val_input = "liver/val_paired_all_hcc_under_50.txt"
#     val_output = "liver/val_paired_all_hcc_under_50_unique.txt"
#     val_unique_count = remove_duplicates(val_input, val_output)
    
#     print(f"训练集去重前的行数: {sum(1 for line in open(train_input) if line.strip() and not line.startswith('...'))}")
#     print(f"训练集去重后的行数: {train_unique_count}")
#     print(f"验证集去重前的行数: {sum(1 for line in open(val_input) if line.strip() and not line.startswith('...'))}")
#     print(f"验证集去重后的行数: {val_unique_count}")

# if __name__ == '__main__':
#     main()
# import nibabel as nib
# import numpy as np
# from pathlib import Path
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# def calculate_ratio(file_path):
#     root = '/mnt/realccvl15/yyang303/ucsf/DiffTumor/logs/registered/'
#     print(file_path)
#     img = nib.load(os.path.join(root, file_path))
#     data = img.get_fdata()
#     # 注意：在分割图像中，肿瘤标签通常为1，肝脏标签为2
#     tumor_voxels = np.sum(data == 2)
#     liver_voxels = np.sum(data == 1)
    
#     ratio = tumor_voxels / liver_voxels if liver_voxels > 0 else 0
#     return ratio, tumor_voxels, liver_voxels

# # 存储所有比率的列表
# pre_ratios = []
# post_ratios = []
# case_ids = []

# # 存储肿瘤体素为0的病例信息
# zero_tumor_cases = []

# # 读取文件列表
# with open('liver/paired_all_hcc_under_50.txt', 'r') as f:
#     lines = f.readlines()

# print("\n=== 详细统计信息 ===")
# # 对每行数据进行处理
# for line in lines:
#     fields = line.strip().split()
#     pre_seg = fields[1]  # 第二个文件
#     post_seg = fields[5]  # 第六个文件
#     case_id = pre_seg.split('/')[1]  # 获取病例ID
#     case_ids.append(case_id)
    
#     pre_ratio, pre_tumor, pre_liver = calculate_ratio(pre_seg)
#     post_ratio, post_tumor, post_liver = calculate_ratio(post_seg)
    
#     # 检查是否有肿瘤体素为0的情况
#     if pre_tumor == 0 or post_tumor == 0:
#         zero_tumor_cases.append({
#             'case_id': pre_seg,
#             'pre_tumor': pre_tumor,
#             'post_tumor': post_tumor,
#             'pre_liver': pre_liver,
#             'post_liver': post_liver
#         })
    
#     pre_ratios.append(pre_ratio)
#     post_ratios.append(post_ratio)
    
#     print(f"\n病例 {case_id}:")
#     print(f"治疗前 - 肿瘤体素: {pre_tumor:.0f}, 肝脏体素: {pre_liver:.0f}, 占比: {pre_ratio*100:.2f}%")
#     print(f"治疗后 - 肿瘤体素: {post_tumor:.0f}, 肝脏体素: {post_liver:.0f}, 占比: {post_ratio*100:.2f}%")

# # 打印肿瘤体素为0的病例统计
# print("\n=== 肿瘤体素为0的病例统计 ===")
# print(f"总计: {len(zero_tumor_cases)}个病例")
# print("\n具体病例:")
# for case in zero_tumor_cases:
#     print(f"病例ID: {case['case_id']}")
#     print(f"治疗前 - 肿瘤体素: {case['pre_tumor']:.0f}, 肝脏体素: {case['pre_liver']:.0f}")
#     print(f"治疗后 - 肿瘤体素: {case['post_tumor']:.0f}, 肝脏体素: {case['post_liver']:.0f}")
#     print("---")

# # 转换为numpy数组并转换为百分比
# pre_ratios = np.array(pre_ratios) * 100
# post_ratios = np.array(post_ratios) * 100

# # 找出治疗前后都小于60%的病例
# # both_under_60 = np.logical_and(pre_ratios < 60, post_ratios < 60)
# # under_60_indices = np.where(both_under_60)[0]


# # 找出治疗前后都小于50%的病例
# # both_under_50 = np.logical_and(pre_ratios < 50, post_ratios < 50)
# # under_50_indices = np.where(both_under_50)[0]

# # 保存符合条件的原始数据行到新文件
# # with open('liver/paired_all_hcc_under_50_val.txt', 'w') as f:
# #     with open('liver/paired_all_hcc_val.txt', 'r') as source:
# #         all_lines = source.readlines()
# #         for idx in under_50_indices:
# #             f.write(all_lines[idx])

# # print(f"\n已将{len(under_50_indices)}个治疗前后比例都小于50%的病例信息保存到 'paired_all_hcc_under_50.txt'")


# # print("\n=== 治疗前后比例都小于50%的病例 ===")
# # print(f"总计: {len(under_50_indices)}个病例")
# # print("\n具体病例:")
# # for idx in under_50_indices:
# #     print(f"病例ID: {case_ids[idx]}")
# #     print(f"治疗前占比: {pre_ratios[idx]:.2f}%")
# #     print(f"治疗后占比: {post_ratios[idx]:.2f}%")
# #     print("---")

# # # 计算变化
# # ratio_changes = post_ratios - pre_ratios
# # print(f"\n治疗前后变化:")
# # print(f"平均变化: {np.mean(ratio_changes):.2f}%")
# # print(f"最大减少: {np.min(ratio_changes):.2f}%")
# # print(f"最大增加: {np.max(ratio_changes):.2f}%")

# # # 统计增加和减少的病例数
# # increased = np.sum(ratio_changes > 0)
# # decreased = np.sum(ratio_changes < 0)
# # unchanged = np.sum(ratio_changes == 0)
# # print(f"\n变化统计:")
# # print(f"增加病例数: {increased}")
# # print(f"减少病例数: {decreased}")
# # print(f"无变化病例数: {unchanged}")

# # # 创建图形
# # plt.figure(figsize=(15, 10))

# # # 1. 治疗前后的分布直方图
# # plt.subplot(2, 2, 1)
# # plt.hist(pre_ratios, bins=10, alpha=0.5, label='治疗前', color='blue')
# # plt.hist(post_ratios, bins=10, alpha=0.5, label='治疗后', color='red')
# # plt.xlabel('肿瘤/肝脏体积比 (%)')
# # plt.ylabel('病例数量')
# # plt.title('治疗前后肿瘤/肝脏比例分布')
# # plt.legend()

# # # 2. 箱线图对比
# # plt.subplot(2, 2, 2)
# # data_to_plot = [pre_ratios, post_ratios]
# # plt.boxplot(data_to_plot, labels=['治疗前', '治疗后'])
# # plt.ylabel('肿瘤/肝脏体积比 (%)')
# # plt.title('治疗前后比例箱线图')

# # # 3. 治疗前后变化的直方图
# # plt.subplot(2, 2, 3)
# # plt.hist(ratio_changes, bins=10, color='green', alpha=0.7)
# # plt.xlabel('比例变化 (%)')
# # plt.ylabel('病例数量')
# # plt.title('治疗前后比例变化分布')

# # # 4. 散点图：治疗前vs治疗后
# # plt.subplot(2, 2, 4)
# # plt.scatter(pre_ratios, post_ratios, alpha=0.6)
# # plt.plot([0, max(max(pre_ratios), max(post_ratios))], 
# #          [0, max(max(pre_ratios), max(post_ratios))], 
# #          'r--', alpha=0.5)  # 对角线
# # plt.xlabel('治疗前比例 (%)')
# # plt.ylabel('治疗后比例 (%)')
# # plt.title('治疗前后比例对比')

# # plt.tight_layout()
# # plt.savefig('ratio_distribution.png')
# # plt.close()

# # # 保持原有的打印统计信息
# # print("\n=== 统计分布 ===")
# # print(f"\n治疗前占比分布:")
# # print(f"最小值: {np.min(pre_ratios):.2f}%")
# # print(f"最大值: {np.max(pre_ratios):.2f}%")
# # print(f"平均值: {np.mean(pre_ratios):.2f}%")
# # print(f"中位数: {np.median(pre_ratios):.2f}%")
# # print(f"标准差: {np.std(pre_ratios):.2f}%")

# # print(f"\n治疗后占比分布:")
# # print(f"最小值: {np.min(post_ratios):.2f}%")
# # print(f"最大值: {np.max(post_ratios):.2f}%")
# # print(f"平均值: {np.mean(post_ratios):.2f}%")
# # print(f"中位数: {np.median(post_ratios):.2f}%")
# # print(f"标准差: {np.std(post_ratios):.2f}%")