import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, r=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.r = r

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.latent_proj = nn.Linear(self.head_dim, r)
        self.output_proj = nn.Linear(r*num_heads, hidden_size)

    def forward(self, hidden_states, **kwargs):
        batch_size, seq_len, _ = hidden_states.size()

        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)

        Q_latent = self.latent_proj(Q)
        K_latent = self.latent_proj(K)
        V_latent = self.latent_proj(V)

        attn_scores = torch.matmul(Q_latent, K_latent.transpose(-1,-2)) / (self.r ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V_latent)

        # Merge heads
        attn_output = attn_output.permute(0,2,1,3).reshape(batch_size, seq_len, self.num_heads*self.r)
        attn_output = self.output_proj(attn_output)

        return attn_output, None