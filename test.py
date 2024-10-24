import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import os
import requests

# 下载Places365类别标签文件
def download_file(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

download_file('https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt',
              'categories_places365.txt')
download_file("http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar", 'resnet50_places365.pth.tar')

# 读取类别标签
with open('categories_places365.txt') as f:
    classes = [line.strip().split(' ')[0][3:] for line in f]

# 优先使用mps (Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device('mps')  #
    print("Using MPS for inference")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA for inference")
else:
    device = torch.device('cpu')
    print("Using CPU for inference")

# 加载预训练的ResNet50模型
model = resnet50(num_classes=365)
checkpoint = torch.load('resnet50_places365.pth.tar', map_location=device, weights_only=True)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义单个视频文件路径
video_file = '/Users/chujiaping/Movies/植入广告数据集/1.mp4'

# 处理视频
video = cv2.VideoCapture(video_file)
frame_count = 0

output_dir = os.path.join('output_frames', os.path.splitext(os.path.basename(video_file))[0])
os.makedirs(output_dir, exist_ok=True)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    # 每隔一定帧数处理一次（可以根据需要调整）
    if frame_count % 5 == 0:
        # 转换为PIL图像并进行预处理
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img_pil).unsqueeze(0).to(device)  # 确保张量和模型都在同一设备上

        # 进行预测
        with torch.no_grad():
            output = model(input_tensor)

        # 获取最高概率的类别
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

        # 可视化结果
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 保存结果
        output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(output_path, frame)

        print(f"Frame {frame_count} saved to {output_path} with scene: {label}")

# 释放视频对象
video.release()
cv2.destroyAllWindows()
