import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据: 构建正弦曲线
# 生成 0 到 10 之间的 400 个点，并增加少量噪声
X_numpy = np.linspace(0, 10, 400).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(400, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成：正弦波数据。")
print("---" * 10)


# 2. 构建多层非线性模型
# 对于复杂的波浪线，我们需要“深度”来提供表达能力
class SinPredictor(nn.Module):
    def __init__(self):
        super(SinPredictor, self).__init__()
        # 这种结构叫 MLP (多层感知机)
        self.net = nn.Sequential(
            nn.Linear(1, 64),  # 第一层：将 1 个输入扩充到 64 个特征
            nn.ReLU(),  # 激活函数：关键！注入“弯曲”能力，没有它模型只能拟合直线
            nn.Linear(64, 64),  # 第二层：隐藏层，进一步提取非线性特征
            nn.ReLU(),  # 再次激活
            nn.Linear(64, 1)  # 输出层：将特征汇总回 1 个预测值
        )

    def forward(self, x):
        return self.net(x)


model = SinPredictor()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
# Adam 优化器在处理这种非线性波动时比 SGD 更稳健
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 可视化预测结果
model.eval()  # 切换到评估模式
with torch.no_grad():
    # 生成更密集的点来绘制平滑的拟合曲线
    X_test = torch.linspace(0, 10, 500).reshape(-1, 1)
    y_test_pred = model(X_test)

plt.figure(figsize=(10, 6))
# 绘制原始带噪声的数据点
plt.scatter(X_numpy, y_numpy, label='Raw data (Sin + Noise)', color='blue', alpha=0.3)
# 绘制模型学习到的曲线
plt.plot(X_test.numpy(), y_test_pred.numpy(), label='Neural Network Fit', color='red', linewidth=3)
plt.title("Using Multi-layer Perceptron to fit Sin Function")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
