import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
import glob
import requests


# 下载类别标签文件
def download_categories():
    url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    response = requests.get(url)
    with open('categories_places365.txt', 'wb') as f:
        f.write(response.content)


# 读取类别标签
def load_classes():
    classes = []
    with open('categories_places365.txt') as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    return classes


# 加载 ResNet152 模型
def load_resnet152():
    model = models.resnet152(pretrained=False, num_classes=365)
    checkpoint = torch.load('resnet152_places365.pth.tar', map_location=torch.device('cpu'))
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


# 定义图像预处理步骤
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0)


# 视频帧处理和场景预测
def process_video(video_folder, output_folder, model, classes):
    video_files = glob.glob(os.path.join(video_folder, '*.mp4'))  # 查找文件夹中的所有.mp4视频文件

    for video_file in video_files:
        video = cv2.VideoCapture(video_file)
        frame_count = 0
        output_dir = os.path.join(output_folder, os.path.splitext(os.path.basename(video_file))[0])
        os.makedirs(output_dir, exist_ok=True)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 30 == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                input_tensor = preprocess_image(img_pil)

                with torch.no_grad():
                    output = model(input_tensor)

                _, predicted = torch.max(output, 1)
                label = classes[predicted.item()]

                # 可视化结果
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 保存结果
                output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
                cv2.imwrite(output_path, frame)

                print(f"Frame {frame_count} of video {video_file} saved to {output_path} with scene: {label}")

        video.release()
    cv2.destroyAllWindows()


# 主程序
if __name__ == '__main__':
    # 下载类别标签文件
    if not os.path.exists('categories_places365.txt'):
        download_categories()

    # 读取类别标签
    classes = load_classes()

    # 加载模型
    model = load_resnet152()

    # 定义视频文件夹路径
    video_folder = '/Users/chujiaping/Movies/植入广告数据集'  # 替换为实际视频文件夹路径
    output_folder = 'output_frames'

    # 处理视频
    process_video(video_folder, output_folder, model, classes)
