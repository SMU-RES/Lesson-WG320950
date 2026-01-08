import tkinter as tk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 引入 functional 模块，通常命名为 F
import torch.nn.functional as F

# ==========================================
# 1. 定义 CNN 模型 (LeNet-5 风格)
# ==========================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # --- 卷积层 1 ---
        # 输入: 1通道 (黑白图), 输出: 10通道 (提取10种特征), 卷积核: 5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # --- 卷积层 2 ---
        # 输入: 10通道, 输出: 20通道, 卷积核: 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # --- Dropout2d ---
        # 专门针对图像的 Dropout，随机丢弃整个通道
        self.conv2_drop = nn.Dropout2d()
        
        # --- 全连接层 ---
        # 经过两次卷积和池化后，图像变小了，我们需要计算一下展平后的维度
        # 28x28 -> conv1(5x5) -> 24x24 -> max_pool(2) -> 12x12
        # 12x12 -> conv2(5x5) -> 8x8   -> max_pool(2) -> 4x4
        # 所以展平后是: 20通道 * 4 * 4 = 320
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 第一层: 卷积 -> 池化 -> ReLU
        # max_pool2d(2) 会把图片长宽减半 (下采样)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # 第二层: 卷积 -> Dropout -> 池化 -> ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # 展平: 把立体的特征图拉平成一维向量
        x = x.view(-1, 320)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        
        # [修复] 使用 F.dropout (torch.nn.functional.dropout)
        # 它的参数是 input, p, training
        x = F.dropout(x, training=self.training, p=0.5)
        
        x = self.fc2(x)
        return x

# ==========================================
# 2. 训练函数
# ==========================================
def train_cnn_model():
    print("正在准备数据 (CNN)...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # 同样合并数据
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("开始训练 CNN (Epochs: 3)...")
    model.train()
    
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1} [{batch_idx * len(data)}/{len(full_dataset)}] Loss: {loss.item():.4f}')

    print("CNN 模型训练完成!")
    model.eval()
    return model, device

# ==========================================
# 3. GUI 类 (复用之前的逻辑)
# ==========================================
class PixelGridGUI:
    def __init__(self, root, model, device):
        self.root = root
        self.root.title("CNN 像素级手写识别")
        self.model = model
        self.device = device
        
        self.grid_size = 28
        self.cell_size = 15
        self.canvas_size = self.grid_size * self.cell_size
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        main_frame = tk.Frame(root)
        main_frame.pack(padx=20, pady=20)

        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, 
                                bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack()
        self._draw_grid_lines()
        
        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack(pady=10)
        
        self.btn_predict = tk.Button(ctrl_frame, text="识别 (Predict)", command=self.predict,
                                     bg="#9C27B0", fg="white", font=("Arial", 12), width=15)
        self.btn_predict.pack(side=tk.LEFT, padx=10)
        
        self.btn_clear = tk.Button(ctrl_frame, text="清除 (Clear)", command=self.clear_grid,
                                   bg="#f44336", fg="white", font=("Arial", 12), width=15)
        self.btn_clear.pack(side=tk.LEFT, padx=10)
        
        self.lbl_result = tk.Label(root, text="Waiting...", font=("Arial", 18, "bold"), fg="#333")
        self.lbl_result.pack(pady=20)
        
        self.canvas.bind("<Button-1>", self.paint_cell)
        self.canvas.bind("<B1-Motion>", self.paint_cell)

    def _draw_grid_lines(self):
        for i in range(self.grid_size):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.canvas_size, fill="#eee")
            y = i * self.cell_size
            self.canvas.create_line(0, y, self.canvas_size, y, fill="#eee")

    def paint_cell(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            self.grid_data[row, col] = 1.0
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="")

    def clear_grid(self):
        self.canvas.delete("all")
        self._draw_grid_lines()
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.lbl_result.config(text="Waiting...")

    def predict(self):
        # CNN 需要完整的 4D 输入 (Batch, Channel, Height, Width)
        # 且不应该像全连接层那样先展平
        tensor_data = torch.from_numpy(self.grid_data).float()
        tensor_data = tensor_data.unsqueeze(0).unsqueeze(0) # (1, 1, 28, 28)
        
        # 归一化
        tensor_data = (tensor_data - 0.1307) / 0.3081
        tensor_data = tensor_data.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_data)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            val = predicted.item()
            conf = confidence.item() * 100

        self.lbl_result.config(text=f"Result: {val}  (Conf: {conf:.1f}%)")

if __name__ == "__main__":
    model, device = train_cnn_model()
    root = tk.Tk()
    root.resizable(False, False)
    app = PixelGridGUI(root, model, device)
    root.mainloop()