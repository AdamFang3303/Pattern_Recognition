import torch
from torch.utils.data import DataLoader, random_split
from config import Config
from data import NPYShapeDataset
from model import EnhancedCNN
from utils import EarlyStopper
from torch import nn
from torch.amp import autocast, GradScaler

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集
    full_dataset = NPYShapeDataset(Config.npy_dir)
    train_size = int(Config.train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 数据加载
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size * 2,
        num_workers=Config.num_workers,
        pin_memory=True
    )

    # 模型初始化
    model = EnhancedCNN(Config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()   # 混合精度
    early_stopper = EarlyStopper(Config.early_stop)

    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        # 验证
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        # 打印统计信息
        train_loss /= len(train_dataset)
        val_loss /= len(test_dataset)
        accuracy = 100 * correct / len(test_dataset)

        print(f"Epoch {epoch + 1}/{Config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.2f}%")

        if early_stopper.should_stop(val_loss):
            print("Early stopping triggered!")
            break

    torch.save(model.state_dict(), "shape_recognition.pth")
    print(f"Training completed in {epoch + 1} epochs. Model saved.")


if __name__ == "__main__":
    train()