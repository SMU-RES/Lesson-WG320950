import tkinter as tk
import numpy as np
from sklearn.svm import SVC
from torchvision import datasets
import time

# ==========================================
# 1. 训练函数 (SVM)
# ==========================================
def train_svm_model():
    print("正在准备数据 (使用 torchvision 加载)...")
    
    # 使用 torchvision 下载/加载数据，避免依赖 tensorflow
    # 我们只用它来获取数据，之后转为 numpy 给 sklearn 使用
    train_dataset = datasets.MNIST('./data', train=True, download=True)
    test_dataset = datasets.MNIST('./data', train=False, download=True)

    # 提取数据 (Tensor/PIL -> Numpy)
    # train_dataset.data 是 (N, 28, 28) 的 tensor (uint8, 0-255)
    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    # 1. 合并所有数据 (70,000 张)
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))

    # 2. 预处理
    # 展平: (N, 28, 28) -> (N, 784)
    n_samples = len(X_all)
    X_flattened = X_all.reshape((n_samples, -1))
    
    # 归一化: 0-255 -> 0.0-1.0
    # SVM 对特征缩放非常敏感，必须归一化
    X_normalized = X_flattened / 255.0

    print(f"数据准备完成。样本总数: {n_samples}")
    print("开始训练 SVM (RBF核)...")
    print("⚠️ 注意: 使用 70,000 张图片训练 SVM 比较耗时 (可能需要 5-10 分钟)，请耐心等待...")
    
    start_time = time.time()
    
    # 创建 SVM 分类器
    # gamma='scale' 是默认值，C=5.0 稍微增加一点正则化惩罚
    # cache_size=1000 允许使用更多内存(MB)来加速训练
    svm = SVC(kernel='rbf', C=5.0, gamma='scale', cache_size=1000)
    
    svm.fit(X_normalized, y_all)
    
    end_time = time.time()
    print(f"训练完成! 耗时: {end_time - start_time:.2f} 秒")
    
    return svm

# ==========================================
# 2. 像素网格 GUI 类 (复用逻辑)
# ==========================================
class PixelGridGUI:
    def __init__(self, root, model):
        self.root = root
        self.root.title("SVM 像素级手写识别")
        self.model = model
        
        # 网格设置
        self.grid_size = 28   # 28x28 格子
        self.cell_size = 15   # 放大显示
        self.canvas_size = self.grid_size * self.cell_size
        
        # 数据存储 (0.0 = 黑/空, 1.0 = 白/笔迹)
        # 输入给 SVM 的数据应该是 0.0 - 1.0 之间的浮点数
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # UI 布局
        main_frame = tk.Frame(root)
        main_frame.pack(padx=20, pady=20)

        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, 
                                bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack()
        
        self._draw_grid_lines()
        
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
        # 1. 准备数据
        # SVM 训练时使用的是 Flatten 后的 (N, 784) 且范围在 0-1 之间
        # grid_data 已经是 (28, 28) 且范围 0-1
        
        # 展平: (1, 784)
        flat_data = self.grid_data.reshape(1, -1)
        
        # 2. 推理
        prediction = self.model.predict(flat_data)
        
        # SVM 默认不输出概率 (为了速度)，如果需要概率需要训练时开启 probability=True (非常慢)
        # 所以这里我们直接显示预测类别
        self.lbl_result.config(text=f"Result: {prediction[0]}")

if __name__ == "__main__":
    # 1. 训练
    model = train_svm_model()
    
    # 2. 启动 GUI
    root = tk.Tk()
    root.resizable(False, False)
    
    app = PixelGridGUI(root, model)
    root.mainloop()