import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EfficientNetTransformer(nn.Module):
    def __init__(
        self, 
        num_classes=10, 
        d_model=1280,
        nhead=8, 
        num_layers=4, 
        dropout=0.3
    ):
        super().__init__()
        
        # Backbone CNN
        print("Initializing EfficientNet-B0 backbone...")
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity() 
        
        self.feature_projection = nn.Linear(1280, d_model) 
        # ------------------------
        
        # Positional Encoding dùng d_model mới
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer Encoder dùng d_model mới
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, # Thường gấp đôi hoặc gấp 4 d_model
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # Trích xuất đặc trưng từ CNN
        x = x.view(b * t, c, h, w)
        features = self.backbone(x) # Luôn ra (b*t, 1280)
        
        # Chuyển đổi về d_model custom của bạn
        features = self.feature_projection(features) # Giờ nó mới ra (b*t, d_model)
        
        features = features.view(b, t, -1)
        
        # Tiếp tục luồng Transformer
        features = self.pos_encoder(features)
        transformer_out = self.transformer_encoder(features)
        
        mean_features = torch.mean(transformer_out, dim=1) 
        logits = self.classifier(mean_features)
        
        return logits
    
def example():
    model = EfficientNetTransformer(
        num_classes=10)
    
    # Input giả lập: Batch=2, Frames=16, C=3, H=224, W=224
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Mong đợi: (2, 10)