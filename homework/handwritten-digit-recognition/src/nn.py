import tkinter as tk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 定义 PyTorch 模型
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # MNIST 图片是 28x28 = 784
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 展平: (Batch_Size, 1, 28, 28) -> (Batch_Size, 784)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x # 输出 Logits，不需要 Softmax，因为 CrossEntropyLoss 会处理

# ==========================================
# 2. 训练函数
# ==========================================
def train_model():
    print("正在准备数据 (使用 PyTorch)...")
    
    # 定义数据转换：转为Tensor -> 归一化
    # MNIST 均值 0.1307，标准差 0.3081 是官方推荐的归一化参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载/加载训练集和测试集
    # PyTorch 的 datasets 能够自动处理下载
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # 合并数据集 (按照要求使用所有数据)
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    
    train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    print(f"数据加载完成，总样本数: {len(full_dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("开始训练 (Epochs: 3)...")
    model.train()
    
    for epoch in range(3):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1} [{batch_idx * len(data)}/{len(full_dataset)}] Loss: {loss.item():.4f}')

    print("模型训练完成!")
    model.eval() # 切换到评估模式
    return model, device

# ==========================================
# 3. 像素网格 GUI 类
# ==========================================
class PixelGridGUI:
    def __init__(self, root, model, device):
        self.root = root
        self.root.title("PyTorch 像素级手写识别")
        self.model = model
        self.device = device
        
        # 网格设置
        self.grid_size = 28   # 28x28 格子
        self.cell_size = 15   # 每个格子在屏幕上的像素大小 (放大显示)
        self.canvas_size = self.grid_size * self.cell_size
        
        # 数据存储 (0.0 = 黑/空, 1.0 = 白/笔迹)
        # 注意：MNIST是黑底白字，但为了绘图习惯，我们显示白底黑字，
        # 在预测时再翻转或调整数值。
        # 这里我们直接存储“墨水浓度”，0为空，1为满。
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # UI 布局
        main_frame = tk.Frame(root)
        main_frame.pack(padx=20, pady=20)

        # 画布
        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, 
                                bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack()
        
        # 绘制网格线
        self._draw_grid_lines()
        
        # 控制面板
        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack(pady=10)
        
        self.btn_predict = tk.Button(ctrl_frame, text="识别 (Predict)", command=self.predict,
                                     bg="#4CAF50", fg="white", font=("Arial", 12), width=15)
        self.btn_predict.pack(side=tk.LEFT, padx=10)
        
        self.btn_clear = tk.Button(ctrl_frame, text="清除 (Clear)", command=self.clear_grid,
                                   bg="#f44336", fg="white", font=("Arial", 12), width=15)
        self.btn_clear.pack(side=tk.LEFT, padx=10)
        
        self.lbl_result = tk.Label(root, text="Waiting...", font=("Arial", 18, "bold"), fg="#333")
        self.lbl_result.pack(pady=20)
        
        # 事件绑定
        self.canvas.bind("<Button-1>", self.paint_cell) # 点击
        self.canvas.bind("<B1-Motion>", self.paint_cell) # 拖拽

    def _draw_grid_lines(self):
        """绘制浅灰色的网格辅助线"""
        for i in range(self.grid_size):
            # 垂直线
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.canvas_size, fill="#eee")
            # 水平线
            y = i * self.cell_size
            self.canvas.create_line(0, y, self.canvas_size, y, fill="#eee")

    def paint_cell(self, event):
        """处理鼠标绘图：计算点击了哪个格子并填充"""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        # 边界检查
        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            # 更新数据 (设为 1.0 表示完全涂黑)
            # 为了更好的鲁棒性，我们可以让笔刷稍微"晕染"一下周围像素，
            # 或者只画单像素。这里为了模拟真实笔迹，稍微增加一点周围像素的灰度。
            self.grid_data[row, col] = 1.0
            
            # 简单地为上下左右增加一点点数值，模拟抗锯齿/笔触宽度（可选）
            # neighbors = [(0,1), (0,-1), (1,0), (-1,0)]
            # for dr, dc in neighbors:
            #     r, c = row+dr, col+dc
            #     if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            #         self.grid_data[r, c] = max(self.grid_data[r, c], 0.5)

            # 在画布上绘制黑色矩形
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            
            # 使用灰色或黑色填充，视觉上反馈给用户
            # color_val = int(255 * (1 - self.grid_data[row, col])) # 反色显示
            # color_hex = f"#{color_val:02x}{color_val:02x}{color_val:02x}"
            
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="")

    def clear_grid(self):
        self.canvas.delete("all")
        self._draw_grid_lines()
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.lbl_result.config(text="Waiting...")

    def predict(self):
        # 1. 准备数据
        # PyTorch 训练时进行了 Normalize((0.1307,), (0.3081,))
        # 输入数据是 0.0(白/背景) 到 1.0(黑/笔迹)
        # 我们的 grid_data 已经是这个格式了 (因为我们画的是"墨水")
        
        tensor_data = torch.from_numpy(self.grid_data).float()
        
        # 增加维度: (28, 28) -> (1, 1, 28, 28)
        tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
        
        # 归一化处理 (手动复现 transforms.Normalize)
        mean = 0.1307
        std = 0.3081
        tensor_data = (tensor_data - mean) / std
        
        tensor_data = tensor_data.to(self.device)

        # 2. 推理
        with torch.no_grad():
            outputs = self.model(tensor_data)
            # 获取概率 (Softmax)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            val = predicted.item()
            conf = confidence.item() * 100

        self.lbl_result.config(text=f"Result: {val}  (Conf: {conf:.1f}%)")

if __name__ == "__main__":
    # 1. 训练
    model, device = train_model()
    
    # 2. 启动 GUI
    root = tk.Tk()
    # 禁止调整窗口大小，保持网格比例
    root.resizable(False, False)
    
    app = PixelGridGUI(root, model, device)
    root.mainloop()