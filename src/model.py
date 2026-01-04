import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
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
        d_model=1280,       # EfficientNet-B0 feature size
        nhead=8,            # Số lượng Attention Heads
        num_layers=4,       # Số lớp Transformer Encoder
        dropout=0.1
    ):
        super().__init__()
        
        #  --- EfficientNet-B0 ---
        print("Initializing EfficientNet-B0 backbone...")
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity() 
        
        # POSITIONAL ENCODING
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # TRANSFORMER ENCODER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=2048, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLASSIFICATION HEAD
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # --- CNN Feature Extraction ---
        x = x.view(b * t, c, h, w)
        # --- EfficientNet
        features = self.backbone(x) # Output: (480, 1280)
        features = features.view(b, t, -1)
        
        # --- Transformer Modeling ---
        features = self.pos_encoder(features)
        transformer_out = self.transformer_encoder(features)
        
        # --- Classification ---
        mean_features = torch.mean(transformer_out, dim=1) # Shape: (16, 1280)
        
        logits = self.classifier(mean_features) # Shape: (16, num_classes)
        
        return logits
    
def example():
    model = EfficientNetTransformer(num_classes=10)
    
    # Input giả lập: Batch=2, Frames=16, C=3, H=224, W=224
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Mong đợi: (2, 10)