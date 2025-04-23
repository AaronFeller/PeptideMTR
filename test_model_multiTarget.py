# test_multiTarget.py
import torch
from model_multiTarget import MoE_model
from transformers import PreTrainedTokenizerFast

def main():
    # --- setup tokenizer and input ---
    tokenizer = PreTrainedTokenizerFast.from_pretrained("aaronfeller/PeptideMTR")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    tokens = tokenizer(smiles)["input_ids"]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
    seq_len = input_tensor.size(1)

    # common model args
    common = dict(
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

    # --- Test 1: MLM mode (num_tasks=0) ---
    model_mlm = MoE_model(**common, num_tasks=0)
    model_mlm.eval()
    with torch.no_grad():
        logits, lb_loss = model_mlm(input_tensor)
    print("MLM mode -- logits shape:", logits.shape, " | lb_loss:", lb_loss.item())
    assert isinstance(logits, torch.Tensor), "logits not a Tensor"
    assert logits.shape == (1, seq_len, common["output_dim"]), \
        f"Expected logits shape (1,{seq_len},{common['output_dim']}), got {logits.shape}"
    assert lb_loss.dim() == 0, "lb_loss should be a scalar"
    print("MLM mode shapes OK\n")

    # --- Test 2: Multi-task mode ---
    head_sizes = [1, 3, 1, 1, 20]  # e.g. one regression head + one 3-class head
    model_mt = MoE_model(**common, num_tasks=len(head_sizes), head_sizes=head_sizes)
    model_mt.eval()
    with torch.no_grad():
        outs, lb_loss_mt = model_mt(input_tensor)
    print("Multi-task mode -- number of heads:", len(outs), " | lb_loss:", lb_loss_mt.item())
    assert isinstance(outs, list), "Expected list of head outputs"
    assert len(outs) == len(head_sizes), \
        f"Expected {len(head_sizes)} heads, got {len(outs)}"
    for idx, (out, hsize) in enumerate(zip(outs, head_sizes)):
        print(f"  Head {idx} output shape: {out.shape} (expected (1, {hsize}))")
        assert out.shape == (1, hsize), \
            f"Head {idx} expected shape (1,{hsize}), got {out.shape}"
    assert lb_loss_mt.dim() == 0, "lb_loss should be a scalar"
    print("Multi-task mode shapes OK")

if __name__ == "__main__":
    main()