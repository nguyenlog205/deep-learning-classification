import torch
import torch.nn as nn
import torchvision.models as models
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    """
    Đây là một block Mamba hoàn chỉnh, tương đương một TransformerEncoderLayer.
    Cấu trúc: Input -> Norm -> Mamba -> Residual Add -> Output
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,    # Dimension of model
            d_state=d_state,    # SSM state expansion factor
            d_conv=d_conv,      # Local convolution width
            expand=expand,      # Block expansion factor
        )

    def forward(self, x):
        # x shape: [Batch, Sequence_Length, d_model]
        # Residual connection: x = x + mamba(norm(x))
        return x + self.mamba(self.norm(x))

class EfficientNetMamba(nn.Module):
    def __init__(
        self, 
        num_classes=10, 
        d_model=512,      # Giảm xuống 512 cho nhẹ, hoặc giữ 1280 tùy GPU
        num_layers=4,     # Số lượng lớp Mamba xếp chồng lên nhau
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.2
    ):
        super().__init__()
        
        # 1. Backbone CNN (EfficientNet-B0)
        print("Initializing EfficientNet-B0 backbone...")
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity() 
        
        # 2. Projection: Chiếu từ 1280 về d_model
        self.feature_projection = nn.Sequential(
            nn.Linear(1280, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 3. Stack các Mamba Blocks (Thay cho TransformerEncoder)
        # Mamba xử lý chuỗi temporal cực nhanh và mượt
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            ) for _ in range(num_layers)
        ])
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, Frames, Channels, Height, Width]
        b, t, c, h, w = x.shape
        
        # --- BƯỚC 1: Spatial Feature Extraction (CNN) ---
        # Gộp Batch và Frames để đưa vào CNN: (B*T, C, H, W)
        x = x.view(b * t, c, h, w)
        
        # Extract features
        features = self.backbone(x) # (B*T, 1280)
        
        # Project về d_model
        features = self.feature_projection(features) # (B*T, d_model)
        
        # --- BƯỚC 2: Temporal Modeling (Mamba) ---
        # Tách lại Batch và Time: (B, T, d_model)
        features = features.view(b, t, -1)
        
        # Chạy qua từng lớp Mamba
        for layer in self.layers:
            features = layer(features)
            
        # --- BƯỚC 3: Aggregation & Classification ---
        # Có 2 cách phổ biến để lấy đặc trưng cuối cùng:
        
        # Cách A: Average Pooling (Lấy trung bình toàn bộ video) -> Ổn định
        mean_features = torch.mean(features, dim=1) 
        
        # Cách B: Last Token (Lấy trạng thái cuối cùng) -> Tốt nếu video có tính nhân quả mạnh
        # last_features = features[:, -1, :] 
        
        logits = self.classifier(mean_features)
        
        return logits

def example():
    # Setup thiết bị
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # Khởi tạo model
    model = EfficientNetMamba(
        num_classes=10, 
        d_model=256,    # Demo size nhỏ
        num_layers=2    # Demo 2 lớp
    ).to(device)
    
    # Input giả lập: Batch=2, Frames=16, C=3, H=224, W=224
    dummy_input = torch.randn(2, 16, 3, 224, 224).to(device)
    
    # Forward pass
    try:
        output = model(dummy_input)
        print(f"✅ Input shape: {dummy_input.shape}")
        print(f"✅ Output shape: {output.shape}") # Mong đợi: (2, 10)
        print("🚀 EfficientNet + Mamba chạy thành công!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    example()