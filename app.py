import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from models.model import SimpleCNN

# 页面标题
st.title("Health AI Demo")

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# 标签映射
label_map = {0: "normal", 1: "abnormal"}

# 上传图片
uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读取图片
    img = Image.open(uploaded_file).convert("RGB")

    # 显示图片
    st.image(img, caption="上传的图片", use_container_width=True)

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)

    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    # 输出结果
    result = label_map[predicted.item()]
    st.subheader(f"预测结果：{result}")