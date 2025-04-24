from torchvision import transforms

class Config:
    # 数据参数
    npy_dir = "./database"
    batch_size = 512       # 减小batch_size提升稳定性
    num_workers = 0        # Windows必须设为0
    train_ratio = 0.8

    # 模型参数
    num_classes = 9         # 9个类别保持不变
    in_channels = 1
    dropout_rate = 0.4      # 适当提高防止过拟合

    # 训练参数
    epochs = 30            # 延长训练轮次
    lr = 1e-3              # 调高学习率
    weight_decay = 1e-4    # 加强权重衰减
    early_stop = 10        # 增加早停耐心值

    # 学习率调度参数
    scheduler_gamma = 0.1  # 学习率衰减系数

    # 优化后的数据增强
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),                  # 减小旋转幅度
        transforms.RandomAffine(0, shear=10),           # 降低仿射变换强度
        transforms.ColorJitter(brightness=0.1),          # 减少颜色抖动
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # 调整随机擦除
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])     # 保持与GUI一致的归一化
    ])