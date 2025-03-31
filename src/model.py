import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Transformer positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Multi-Head Self-Attention with improved features for NILM
class NILMSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(NILMSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Additional feature enhancement layer
        self.feature_enhancement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Apply multi-head attention
        attn_output, attn_weights = self.mha(x, x, x, attn_mask=mask)
        
        # Apply feature enhancement
        enhanced = self.feature_enhancement(attn_output)
        
        return enhanced + attn_output, attn_weights

# Feed-Forward Network with residual connection
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Using GELU instead of ReLU for better performance
        
    def forward(self, x):
        # Apply FFN with residual connection
        return self.linear2(self.dropout(self.activation(self.linear1(x)))) + x

# Transformer Encoder Layer with modifications for NILM
class NILMEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(NILMEncoderLayer, self).__init__()
        self.self_attn = NILMSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Apply normalization first (Pre-LN Transformer)
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, mask)
        
        # First residual connection
        x = x + self.dropout(attn_output)
        
        # Feed-forward with normalization
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        
        # Second residual connection
        return x + self.dropout(ff_output)

# Complete Transformer-based NILM model (MATNilm-inspired)
class TransformerNILM(nn.Module):
    def __init__(
        self, 
        input_dim=1, 
        output_dim=1, 
        d_model=128, 
        nhead=8, 
        num_encoder_layers=6,
        dim_feedforward=512, 
        dropout=0.1, 
        seq_length=299
    ):
        super(TransformerNILM, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length=seq_length)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            NILMEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Additional attention-guided temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Adaptive power pattern detection
        self.pattern_detection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters with appropriate initialization schemes"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # Input shape: [batch_size, seq_length, input_dim]
        
        # Embed input
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Apply temporal convolution
        x_conv = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # Combine with attention features
        pattern_weights = self.pattern_detection(x)
        x = x * pattern_weights + x_conv * (1 - pattern_weights)
        
        # Final output projection
        output = self.output_projection(x)
        
        return output

# Loss function for NILM
class NILMLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(NILMLoss, self).__init__()
        self.alpha = alpha  # Weight for MAE
        self.beta = beta    # Weight for pattern matching
        self.mse_loss = nn.MSELoss(reduction="mean")
        
    def forward(self, pred, target):
        # Mean Absolute Error
        mae_loss = F.l1_loss(pred, target, reduction="mean")
        
        # Mean Squared Error
        mse_loss = self.mse_loss(pred, target)
        
        # Gradient loss to better capture transitions
        # Calculate gradients along time dimension
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        gradient_loss = F.l1_loss(pred_diff, target_diff, reduction="mean")
        
        # Combined loss
        combined_loss = (self.alpha * mae_loss) + ((1 - self.alpha) * mse_loss) + (self.beta * gradient_loss)
        
        return combined_loss, mae_loss, mse_loss, gradient_loss
