import cv2
import numpy as np
import torch
from model import EnhancedCNN
from config import Config
from data import NPYShapeDataset
import random

# 配置参数
WIN_SIZE = 800
CANVAS_SIZE = 600
BUTTONS = {
    "predict": (610, 20, 180, 40),
    "random_sample": (610, 80, 180, 40),
    "clear": (610, 140, 180, 40)
}

# 类别映射
SHAPE_CLASSES = ['circle', 'square', 'triangle', 'cat', 'dog',
                 'hexagon', 'octagon', 'line', 'star']

# 全局变量
drawing = False
last_point = (-1, -1)
model = None
dataset = None


def create_gui():
    """创建初始画板"""
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    return canvas


def preprocess_image(img):
    """优化图像预处理流程"""
    # 颜色反转：白底黑字 -> 黑底白字
    gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 镜像翻转
    flipped = cv2.flip(gray, 1)

    # 二值化处理
    _, thresh = cv2.threshold(flipped, 127, 255, cv2.THRESH_BINARY)

    # 调整尺寸匹配模型输入
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

    # 转换为张量并归一化
    tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5  # 匹配训练时的Normalize

    return tensor.unsqueeze(0)  # 添加batch维度


def draw_on_canvas(event, x, y, flags, param):
    """鼠标回调函数（已修复按钮区域交互）"""
    global drawing, last_point, canvas

    # 处理右侧按钮区域点击
    if x >= CANVAS_SIZE:
        if event == cv2.EVENT_LBUTTONDOWN:
            handle_button_click(x, y)
        return

    # 处理左侧画布区域绘制
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and last_point != (-1, -1):
            cv2.line(canvas, last_point, (x, y), (0, 0, 0), 15)
            last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = (-1, -1)


def show_prediction(img, pred, prob):
    """显示预测结果"""
    # 清除之前的预测结果
    img[:CANVAS_SIZE, CANVAS_SIZE:] = 240

    # 显示新的预测结果
    cv2.putText(img, f"Prediction: {pred}",
                (20, CANVAS_SIZE + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, f"Confidence: {prob:.1%}",
                (20, CANVAS_SIZE + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Shape Recognizer", img)


def handle_buttons(x, y):
    """判断按钮点击"""
    for btn, (bx, by, bw, bh) in BUTTONS.items():
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return btn
    return None


def handle_button_click(x, y):
    """处理按钮点击事件"""
    global canvas
    btn = handle_buttons(x, y)

    if btn == 'predict':
        # 清除之前的预测结果
        full_img[:CANVAS_SIZE, CANVAS_SIZE:] = 240

        input_tensor = preprocess_image(canvas)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        show_prediction(full_img, SHAPE_CLASSES[pred.item()], conf.item())

    elif btn == 'random_sample':
        # 清除之前的预测结果和真实标签
        full_img[:CANVAS_SIZE, CANVAS_SIZE:] = 240

        sample_img, true_label = load_random_sample()
        canvas[:] = sample_img
        # 显示真实标签
        cv2.putText(full_img, f"True: {SHAPE_CLASSES[true_label]}",
                    (20, CANVAS_SIZE + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
        cv2.imshow("Shape Recognizer", full_img)

    elif btn == 'clear':
        # 清除画布和预测结果区域
        canvas = create_gui()
        full_img[:CANVAS_SIZE, CANVAS_SIZE:] = 240


def load_random_sample():
    """加载随机样本"""
    idx = random.randint(0, len(dataset) - 1)
    sample, label = dataset[idx]
    img = (sample.squeeze().numpy() * 255).astype(np.uint8)
    img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), label.item()


if __name__ == "__main__":
    # 初始化模型
    model = EnhancedCNN(Config)
    model.load_state_dict(torch.load('shape_recognition.pth', map_location='cpu'))
    model.eval()

    # 加载数据集用于随机样本
    dataset = NPYShapeDataset(Config.npy_dir)

    # 创建界面
    canvas = create_gui()
    full_img = np.ones((WIN_SIZE, WIN_SIZE, 3), dtype=np.uint8) * 255
    cv2.namedWindow('Shape Recognizer')
    cv2.setMouseCallback('Shape Recognizer', draw_on_canvas)

    while True:
        # 实时检测窗口关闭状态
        if cv2.getWindowProperty('Shape Recognizer', cv2.WND_PROP_VISIBLE) < 1:
            break

        # 绘制界面元素
        full_img[:CANVAS_SIZE, :CANVAS_SIZE] = canvas
        full_img[:, CANVAS_SIZE:] = 240  # 右侧功能区

        # 绘制按钮
        for btn, (x, y, w, h) in BUTTONS.items():
            cv2.rectangle(full_img, (x, y), (x + w, y + h), (150, 150, 250), -1)
            cv2.putText(full_img, btn.replace('_', ' ').title(),
                        (x + 10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('Shape Recognizer', full_img)

        # 退出检测（支持Q键和窗口关闭按钮）
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()