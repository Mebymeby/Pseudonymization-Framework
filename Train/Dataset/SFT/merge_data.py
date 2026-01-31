import json
import os

# 将两个json文件合并
file1 = "Train/Dataset/SFT/xsum_train_5000_ner_rand_direct.json"
file2 = "Train/Dataset/SFT/cnn_dailymail_train_1000_ner_rand_direct.json"

# 读取文件
with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# 合并数据
merged_data = data1 + data2

# 保存合并后的文件
save_path = os.path.join("Train", "Dataset", "SFT", "merged.json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)
