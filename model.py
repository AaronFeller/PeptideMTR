import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        # x: (N, input_dim)
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        output = self.linear2(F.silu(x1) * x2)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, input_pos=None):
        B, T, C = x.shape  # Batch, sequence, embedding dim
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply rotary positional embeddings to queries and keys
        q = self.rotary(q, input_pos=input_pos)
        k = self.rotary(k, input_pos=input_pos)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


class MoE(nn.Module):
    def __init__(self, embed_dim, ffn_hidden_dim, num_experts, top_k, bias_update_rate=0.01):
        super().__init__()
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate

        self.experts = nn.ModuleList([Expert(embed_dim, ffn_hidden_dim) for _ in range(num_experts)])
        self.expert_biases = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

    def forward(self, x):
        B, T, D = x.shape

        gate_scores = self.gate(x) + self.expert_biases  # (B, T, E)
        topk_vals, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        # print(f"topk_indices: {topk_indices}")
        topk_gates = F.softmax(topk_vals, dim=-1)

        expert_outputs = torch.zeros_like(x)
        current_token_counts = torch.zeros(self.num_experts, device=x.device)

        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx)
            if not mask.any():
                continue
            
            b, t, k = mask.nonzero(as_tuple=True)
            tokens = x[b, t, :]
            gate_weights = topk_gates[b, t, k].unsqueeze(-1)
            expert_out = self.experts[expert_idx](tokens)
            expert_outputs[b, t, :] += gate_weights * expert_out
            current_token_counts[expert_idx] += tokens.size(0)

        # Load balancing loss (GShard-style)
        expert_usage = F.one_hot(topk_indices, num_classes=self.num_experts).float().sum(dim=(0, 1, 2))
        expert_probs = F.softmax(gate_scores, dim=-1).mean(dim=(0, 1))
        usage_ratio = expert_usage / expert_usage.sum()
        prob_ratio = expert_probs / expert_probs.sum()
        load_balancing_loss = (usage_ratio * prob_ratio).sum() * self.num_experts

        if self.training:
            avg_tokens = current_token_counts.mean()
            load_violation = current_token_counts - avg_tokens
            self.expert_biases += self.bias_update_rate * load_violation

        return expert_outputs, load_balancing_loss


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, num_experts, top_k, max_seq_len):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, max_seq_len)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = MoE(embed_dim, ffn_hidden_dim, num_experts, top_k)

    def forward(self, x, input_pos=None):
        x = x + self.attn(self.attn_norm(x), input_pos=input_pos)
        ffn_out, lb_loss = self.ffn(self.ffn_norm(x))  # unpack first
        x = x + ffn_out  # then add
        return x, lb_loss


class TransformerStack(nn.Module):
    def __init__(self, num_blocks, embed_dim, num_heads, ffn_hidden_dim, num_experts, top_k, max_seq_len):
        super().__init__()        
        self.blocks = nn.ModuleList([
            UnifiedTransformerBlock(embed_dim, num_heads, ffn_hidden_dim, num_experts, top_k, max_seq_len)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, input_pos=None):
        total_lb_loss = 0
        for i, block in enumerate(self.blocks):
            # print block number
            # print(f"Processing block {i+1}/{len(self.blocks)}")
            x, lb_loss = block(x, input_pos=input_pos)
            total_lb_loss += lb_loss
        return self.norm(x), total_lb_loss


class MoE_model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_blocks, num_heads, ffn_hidden_dim, num_experts, top_k, output_dim, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = TransformerStack(num_blocks, embed_dim, num_heads, ffn_hidden_dim, num_experts, top_k, max_seq_len)
        self.sequence_head = nn.Linear(embed_dim, output_dim, bias=True)

    def forward(self, x, input_pos=None):
        x = self.embed(x)
        x, lb_loss = self.transformer(x, input_pos=input_pos)
        return self.sequence_head(x), lb_loss
