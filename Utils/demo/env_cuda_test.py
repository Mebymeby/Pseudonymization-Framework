import torch  # 如果pytorch安装成功即可导入

print("cuda是否可用:",torch.cuda.is_available())  # 查看CUDA是否可用
print("cuda可用数量:",torch.cuda.device_count())  # 查看可用的CUDA数量
print("cuda驱动版本:",torch.version.cuda)  # 查看CUDA的版本号
print("cuda默认编号:",torch.cuda.current_device())  # 查看当前使用的CUDA设备
print("cuda设备名称:",torch.cuda.get_device_name(0))  # 查看当前使用的CUDA设备的名称
