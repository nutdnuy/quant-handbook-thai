"""
บทวิเคราะห์โมเดล Deep Learning อนุกรมเวลา (Time-series) ความถี่รายวันของหุ้น A-Share (สถาปัตยกรรมผสม 1D-CNN + GRU)
สาธิต: Forward Pass และ Training Loop ของ CNN/GRU บนเทนเซอร์ 3 มิติสังเคราะห์; ไม่ใช่การทดลองวัดประสิทธิภาพของโมเดล
"""

from typing import Tuple
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class AshareTemporalCNNGRU(nn.Module):
    """
    โครงข่ายผสม Temporal Convolution + GRU (Gated Recurrent Unit) ขนาด Lightweight
    ขนาด Input: (Batch_size, Sequence_length, N_features)
    ขนาด Output: (Batch_size, 1) ทำนายผลตอบแทน
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 20,
        cnn_channels: int = 32,
        gru_hidden: int = 32,
        dropout: float = 0.30
    ):
        super().__init__()
        self.seq_len = seq_len

        # 1. เลเยอร์ 1D Convolution: สกัดรูปแบบจุลภาคของราคาและปริมาณการซื้อขายระยะสั้นเฉพาะจุด (เช่น รูปแบบแท่งเทียนต่อเนื่อง 3 วัน)
        # Conv1d คาดหวัง Input Shape: (Batch, Channels/Features, Seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()

        # 2. GRU: ดักจับความสัมพันธ์ข้ามเวลาในระยะกลางและการจดจำ/ลืมข้อมูล
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )

        # 3. Regularization และส่วนหัวทำนายแบบ Fully Connected
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape ของ x: (B, T, F) -> Transpose เป็น (B, F, T) สำหรับใช้งานใน Conv1d
        x_trans = x.transpose(1, 2)
        conv_out = self.relu(self.bn1(self.conv1(x_trans)))

        # Transpose กลับเป็น (B, T, Channels) ให้ GRU ประมวลผล
        gru_in = conv_out.transpose(1, 2)
        gru_out, _ = self.gru(gru_in)

        # ดึง Hidden State ของ Time Step สุดท้ายในลำดับ
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        # Output ค่าสเกลาร์ที่ทำนายได้
        pred = self.head(last_hidden)
        return pred.squeeze(-1)


# ================= การสาธิตการเทรนด้วยเทนเซอร์สังเคราะห์ =================
if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)

    # พารามิเตอร์ Input จำลอง
    batch_size = 64
    seq_len = 20     # หน้าต่างประวัติราคาและปริมาณการซื้อขายย้อนหลัง 20 วันทำการ
    n_features = 6   # 6 ตัวชี้วัดพื้นฐานด้านราคาและปริมาณ (Open, High, Low, Close, Volume, Amount)
    n_samples = 2000

    # สร้างเทนเซอร์ 3 มิติของอนุกรมเวลา: (จำนวนตัวอย่าง, หน้าต่าง 20 วัน, 6 ฟีเจอร์)
    X_sim = np.random.normal(0, 1.0, (n_samples, seq_len, n_features)).astype(np.float32)
    
    # จำลองสัญญาณจริงที่อ่อนมาก (มีโมเมนตัม 3 วันล่าสุดเป็นตัวขับเคลื่อนหลักแบบเบาบาง)
    true_signal = (X_sim[:, -1, 0] - X_sim[:, -3, 0]) * 0.006
    y_sim = (true_signal + np.random.normal(0, 0.02, n_samples)).astype(np.float32)

    # แบ่งข้อมูล Train 80%, Validation 20%
    split_idx = int(n_samples * 0.8)
    train_ds = TensorDataset(torch.from_numpy(X_sim[:split_idx]), torch.from_numpy(y_sim[:split_idx]))
    val_ds = TensorDataset(torch.from_numpy(X_sim[split_idx:]), torch.from_numpy(y_sim[split_idx:]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AshareTemporalCNNGRU(n_features=n_features, seq_len=seq_len)
    criterion = nn.HuberLoss() # ใช้ Robust Huber Loss ป้องกันผลกระทบจากค่ากระโดดผิดปกติ
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)

    print(">>> 开始训练 1D-CNN + GRU 神经网络 (观察训练集 vs 验证集误差演变)...")
    for epoch in range(1, 6):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                preds = model(bx)
                val_loss += criterion(preds, by).item() * len(by)
        val_loss /= len(val_ds)

        print(f"  Epoch {epoch:2d} | 训练集 Loss: {train_loss:.6f} | 验证集 Loss: {val_loss:.6f}")

    print("\n组件训练演示完成。本例没有真实股票滑窗、GBDT 对照或样本外收益验证，不能据此判断模型优劣。")
