import torch
import torch.nn as nn

class ParallelLSTMWithAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout=0.2,
        num_realizations=10,
        realization_emb_dim=16,
        use_multihead=True,
        num_heads=4,
        fusion_method='concat'  # options: 'concat', 'add', 'gated'
    ):
        super().__init__()
        self.realization_embedding = nn.Embedding(num_realizations, realization_emb_dim)
        self.fusion_method = fusion_method
        self.hidden_size = hidden_size
        # LSTM Branch
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            ) for i in range(num_layers)
        ])
        self.lstm_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        # Attention Branch
        self.attn_input = nn.Linear(input_size, hidden_size)
        if use_multihead:
            self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        else:
            self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        # Learnable Basis Projection Branch
        self.num_bases = 16
        self.basis_proj = nn.Linear(input_size, self.num_bases)
        self.basis_decoder = nn.Linear(self.num_bases, hidden_size)
        # Fusion
        fusion_input_size = hidden_size * 3 if fusion_method == 'concat' else hidden_size
        if fusion_method == 'gated':
            self.gate_fc = nn.Linear(hidden_size * 2, hidden_size)
            self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(fusion_input_size + realization_emb_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x, realization_ids):
        # LSTM branch
        lstm_out = x
        for lstm, norm in zip(self.lstm_layers, self.lstm_norms):
            residual = lstm_out
            lstm_out, _ = lstm(lstm_out)
            lstm_out = norm(lstm_out + residual)
        lstm_out = lstm_out[:, -1, :]  # Last step
        # Attention branch
        attn_input = self.attn_input(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        attn_out = attn_out.max(dim=1).values
        # Learnable Basis Projection branch
        basis_coeffs = self.basis_proj(x).max(dim=1).values  # [B, num_bases]
        basis_out = self.basis_decoder(basis_coeffs)   # [B, H]
        # Fusion
        if self.fusion_method == 'add':
            fused = lstm_out + attn_out + basis_out
        elif self.fusion_method == 'gated':
            gate_attn = self.sigmoid(self.gate_fc(torch.cat([lstm_out, attn_out], dim=1)))
            fused_attn = gate_attn * lstm_out + (1 - gate_attn) * attn_out
            fused = fused_attn + basis_out
        else:  # concat
            fused = torch.cat([lstm_out, attn_out, basis_out], dim=1)
        realization_emb = self.realization_embedding(realization_ids)
        combined = torch.cat([fused, realization_emb], dim=1)
        x = self.fc1(combined)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
