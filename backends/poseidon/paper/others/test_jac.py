import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
import torchvision.models as models

# 1. 加载ResNet18并定位目标block
resnet = models.resnet18(pretrained=False)
target_block = resnet.layer2[0]  # 目标block：layer2的第一个BasicBlock

# 2. 注册前向钩子，捕获目标block的输入
block_input = None
def forward_hook(module, input, output):
    global block_input
    block_input = input[0].detach().clone()  # 保存目标block的输入（含批量维度）

hook_handle = target_block.register_forward_hook(forward_hook)

# 3. 准备输入并捕获目标block的输入
B = 256  # 批量大小
C, H, W = 3, 224, 224
input_tensor = torch.randn(B, C, H, W)
output_tensor = resnet(input_tensor)  # 触发前向传播，获取block_input
hook_handle.remove()

# 4. 定义从目标block输入到输出的映射函数
def block_input_to_output(x):
    current = target_block(x)  # 目标block的前向传播
    # 处理layer2中剩余的block
    layer2 = resnet.layer2
    target_idx = list(layer2.children()).index(target_block)
    for block in list(layer2.children())[target_idx + 1:]:
        current = block(current)
    # 后续层传播
    current = resnet.layer3(current)
    current = resnet.layer4(current)
    current = resnet.avgpool(current)
    current = torch.flatten(current, 1)
    current = resnet.fc(current)
    return current

# 5. 计算原始雅可比矩阵（含冗余批量维度）
jac = jacobian(
    func=block_input_to_output,
    inputs=block_input.requires_grad_(),
    create_graph=False
)
print("Origin Jacobian(dY/dZ) Shape:", jac.shape)  # [B, 1000, B, 128, 56, 56]

print("Target Tensor(Z) Shape:",block_input.shape)
print("Output Tennsor(Y) Shape", output_tensor.shape)

# # 6. 简化批量维度（提取有效部分：输出样本i对应输入样本i）
# # 生成匹配索引：输出批量i对应输入批量i
# batch_indices = torch.arange(B)  # 形状：[B]
# jac_simplified = jac[batch_indices, :, batch_indices, :, :, :]  # 核心切片操作

# print("简化后雅可比矩阵尺寸:", jac_simplified.shape)  # [B, 1000, 128, 56, 56]