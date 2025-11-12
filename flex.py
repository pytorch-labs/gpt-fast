# flex.py
import torch
from torch import nn

@torch.no_grad()
def speculative_generate(
    target_model: nn.Module,
    draft_model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    rng_seed: int | None = 0,
):
    """
    Deterministic speculative decoding that matches baseline greedy when temperature == 0.
    When temperature > 0, behaves like stochastic speculative decoding.
    """

    device = input_ids.device
    torch.manual_seed(rng_seed if rng_seed is not None else 0)

    # Put both models on same device & in eval mode
    target_model.to(device).eval()
    draft_model.to(device).eval()

    seq = input_ids.clone()
    generated = []

    for _ in range(max_new_tokens):
        # -------------------- Draft proposes --------------------
        with torch.no_grad():
            logits_d = draft_model(seq)
            next_token_logits = logits_d[:, -1, :]

            if temperature == 0.0:
                draft_token = torch.argmax(next_token_logits, dim=-1)
            else:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative_probs > top_p
                    sorted_probs[cutoff] = 0.0
                    probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                draft_token = torch.multinomial(probs, 1).squeeze(-1)

        seq_draft = torch.cat([seq, draft_token.unsqueeze(1)], dim=1)

        # -------------------- Target verifies --------------------
        with torch.no_grad():
            logits_t = target_model(seq)
            target_next_logits = logits_t[:, -1, :]

            if temperature == 0.0:
                target_token = torch.argmax(target_next_logits, dim=-1)
            else:
                probs_t = torch.softmax(target_next_logits / temperature, dim=-1)
                target_token = torch.multinomial(probs_t, 1).squeeze(-1)

        # -------------------- Accept or reject --------------------
        if target_token.item() == draft_token.item():
            # accept
            seq = seq_draft
            generated.append(target_token.item())
        else:
            # reject draft; append target token
            seq = torch.cat([seq, target_token.unsqueeze(1)], dim=1)
            generated.append(target_token.item())

        # -------------------- Stop on EOS --------------------
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break

    return seq, torch.tensor(generated, device=device)
