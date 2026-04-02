import torch
from models.model import SimpleCNN
from PIL import Image
import torchvision.transforms as transforms

def predict():
    # 1. 创建模型
    model = SimpleCNN()

    # 2. 加载训练好的参数
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # 3. 读取图片
    img = Image.open("data/test.jpg").convert("RGB")

    # 4. 预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    img = transform(img)
    img = img.unsqueeze(0)

    # 5. 模型推理
    output = model(img)
    print("模型输出：", output)

    # 6. 取预测类别
    _, predicted = torch.max(output, 1)

    label_map = {0: "normal", 1: "abnormal"}
    print("预测类别：", label_map[predicted.item()])

if __name__ == "__main__":
    predict()