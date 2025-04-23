# run_moe_model.py
import torch
from model_multiTarget import MoE_model  # assume your model is saved as model.py
from transformers import PreTrainedTokenizerFast

# Function to encode SMILES strings
tokenizer = PreTrainedTokenizerFast.from_pretrained("aaronfeller/PeptideMTR")

# Define an example SMILES string
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

print("Original SMILES:", smiles)
# Convert SMILES to tokens using your encoder
tokens = tokenizer(smiles)["input_ids"]  # should return a list/array of ints
print("Tokens:", tokens)

input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
print("Input tensor shape:", input_tensor.shape)

# Instantiate the model (same config as before)
model = MoE_model(
    vocab_size=405,
    embed_dim=320,
    num_blocks=6,
    num_heads=8,
    ffn_hidden_dim=1280,
    num_experts=8,
    top_k=2,
    output_dim=405,
    max_seq_len=2048,
)

print("Model instantiated.")

# === Dummy training loop for BERT‐style MLM + MoE load balancing ===
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=-100)
lambda_lb = 0.01
mask_prob = 0.15

model.train()
for epoch in range(10):
    optimizer.zero_grad()

    # create mask for MLM
    mask = torch.rand(input_tensor.shape, device=input_tensor.device) < mask_prob
    input_masked = input_tensor.clone()
    input_masked[mask] = tokenizer.mask_token_id

    # prepare labels: only compute loss on masked tokens
    labels = input_tensor.clone()
    labels[~mask] = -100

    # forward
    logits, lb_loss = model(input_masked)
    logits = logits.view(-1, logits.size(-1))    # (B*T, V)
    labels = labels.view(-1)                     # (B*T,)

    # MLM loss + load-balancing loss
    loss_task = loss_fn(logits, labels)
    loss = loss_task + lambda_lb * lb_loss

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, MLM_loss={loss_task.item():.4f}, lb_loss={lb_loss.item():.4f}, total={loss.item():.4f}")


print("Training complete. Checking final expert selection:")

model.eval()
with torch.no_grad():
    # Grab the first MoE block to inspect its gating
    block = model.transformer.blocks[0]

    # Recompute its input exactly as in forward():
    x_emb  = model.embed(input_tensor)
    attn_o = block.attn(block.attn_norm(x_emb))
    residual = x_emb + attn_o
    ffn_in = block.ffn_norm(residual)

    # Compute gate scores and top-k
    gate_scores    = block.ffn.gate(ffn_in) + block.ffn.expert_biases
    _, topk_indices = torch.topk(gate_scores, block.ffn.top_k, dim=-1)

    print("Top-k expert indices for each token:", topk_indices)


# # Run the model
# model.eval()
# with torch.no_grad():
#     output, lb_loss = model(input_tensor)
#     print("Output shape:", output.shape)
#     print("lb_loss:", lb_loss)

