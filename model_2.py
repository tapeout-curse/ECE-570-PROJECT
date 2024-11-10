class EnhancedTCMModel(nn.Module):
    def __init__(self, input_dim=256, num_heads=4, num_layers=2):
        super(EnhancedTCMModel, self).__init__()

        # Multi-head Attention in encoding path for refined feature extraction
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

        # Encoder and Decoder Transformer layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Additional Adaptive Quantization Layer
        self.quantization = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)

        # Extended Residual Connections in TCM block
        self.residual_connection = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=1)
        )

    def forward(self, x):
        # Multi-head Attention applied in encoding path
        x = x.view(-1, x.size(2), x.size(3))  # Flatten for attention
        x, _ = self.attention(x, x, x)

        # Transformer encoding and decoding
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)

        # Adaptive Quantization and Residual Connection
        quantized = self.quantization(decoded.view(decoded.size(0), -1, 128, 128))
        decoded_with_residual = quantized + self.residual_connection(quantized)

        return decoded_with_residual
