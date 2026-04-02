import torch
import torch.nn as nn
import torch.optim as optim

from models.model import SimpleCNN

def train():
    # 1. 创建模型
    model = SimpleCNN()

    # 2. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. 生成假数据
    # 假设有 20 张图片，每张是 3×64×64
    fake_images = torch.randn(20, 3, 64, 64)

    # 假设标签是 0 或 1
    fake_labels = torch.randint(0, 2, (20,))

    # 4. 训练 5 轮
    for epoch in range(5):
        optimizer.zero_grad()

        outputs = model(fake_images)
        loss = criterion(outputs, fake_labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("模型已保存！")

if __name__ == "__main__":
    train()