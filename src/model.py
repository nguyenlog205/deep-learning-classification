import torch
import torch.nn as nn
import timm

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM)
    (Same as before: Shift left/right, zero parameters)
    """
    def __init__(self, n_segment=8, n_div=8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold] 
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold] 
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:] 
        return out.view(nt, c, h, w)

class TSMBlock(nn.Module):
    """
    Wraps an existing block (e.g., MBConv) with Temporal Shift.
    """
    def __init__(self, net_block, n_segment):
        super(TSMBlock, self).__init__()
        self.tsm = TemporalShift(n_segment=n_segment)
        self.net_block = net_block

    def forward(self, x):
        x = self.tsm(x)
        return self.net_block(x)

class ActionHead(nn.Module):
    """
    Custom MLP Head for Classification.
    Structure: Dropout -> Linear -> Hardswish -> Dropout -> Linear
    """
    def __init__(self, in_features, hidden_dim, num_classes, dropout_p=0.5):
        super(ActionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, hidden_dim),
            nn.Hardswish(),  # Efficient activation function
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class EfficientNetTSM(nn.Module):
    def __init__(self, num_classes=10, n_segments=8, pretrained=True, dropout_p=0.5):
        super(EfficientNetTSM, self).__init__()
        self.n_segments = n_segments
        
        # 1. Load Backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, drop_path_rate=0.2)
        
        # 2. Inject TSM
        self._inject_tsm()
        
        # 3. Define Custom Head
        # EfficientNet-B0 usually has 1280 output features
        num_features = self.backbone.classifier.in_features
        
        # Remove original classifier to save memory (optional)
        self.backbone.classifier = nn.Identity()
        
        # Add robust head
        self.head = ActionHead(
            in_features=num_features, 
            hidden_dim=512,           # Compress to 512
            num_classes=num_classes, 
            dropout_p=dropout_p
        )

    def _inject_tsm(self):
        for stage_idx, stage in enumerate(self.backbone.blocks):
            for block_idx, block in enumerate(stage):
                stage[block_idx] = TSMBlock(block, self.n_segments)

    def forward(self, x):
        # Input: (Batch, Frames, C, H, W)
        b, t, c, h, w = x.size()
        
        # Flatten for backbone
        x = x.view(b * t, c, h, w)
        
        # Backbone Features
        features = self.backbone.forward_features(x)
        
        # Global Pooling
        features = self.backbone.global_pool(features) 
        if isinstance(features, tuple): features = features[0]
        features = features.flatten(1) # (B*T, 1280)
        
        # Temporal Consensus (Average)
        features = features.view(b, t, -1)
        consensus = features.mean(dim=1) # (B, 1280)
        
        # Classification Head
        logits = self.head(consensus) # (B, Num_Classes)
        
        return logits

def example():
    model = EfficientNetTSM(num_classes=10, n_segments=8, dropout_p=0.5)
    dummy_input = torch.randn(2, 8, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}") # (2, 10)
