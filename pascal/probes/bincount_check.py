"""Verify the PyTorch bincount substitution against a naive reference.

_bincount_torch replaces a Triton kernel that cannot be assembled for sm_61, so
it has to be checked against something independent rather than against the
kernel it replaces (which is exactly what will not run here).

The reference below is a direct transcription of _bincount_kernel's semantics in
plain Python loops: for each selected request, tokens in [0, prompt_len) set a
presence bit, and tokens in [prompt_len, prefill_len) increment a count.

    python pascal/probes/bincount_check.py
"""

from __future__ import annotations

import sys

import torch


def reference(
    rows: list[int],
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    vocab_size: int,
    num_words: int,
):
    mask = torch.zeros(len(rows), num_words, dtype=torch.int32)
    counts = torch.zeros(len(rows), vocab_size, dtype=torch.int32)
    for i, r in enumerate(rows):
        plen = int(prompt_len[r])
        flen = int(prefill_len[r])
        for pos in range(flen):
            tok = int(all_token_ids[r, pos])
            if pos < plen:
                mask[i, tok // 32] |= 1 << (tok % 32)
            else:
                counts[i, tok] += 1
    return mask, counts


def main() -> int:
    from vllm.v1.worker.gpu.sample.penalties import _bincount_torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    max_reqs, max_len, vocab_size = 8, 64, 257  # deliberately not a multiple of 32
    num_words = (vocab_size + 31) // 32

    all_token_ids = torch.randint(0, vocab_size, (max_reqs, max_len), device=device, dtype=torch.int32)
    # Force duplicates, and specifically token 0, which is where a naive
    # implementation using index 0 as its masked-out sentinel would go wrong.
    all_token_ids[:, :4] = 0
    all_token_ids[:, 4:8] = 5

    prompt_len = torch.tensor([10, 0, 32, 7, 20, 1, 64, 15], device=device, dtype=torch.int32)
    prefill_len = torch.tensor([40, 16, 48, 7, 33, 9, 64, 60], device=device, dtype=torch.int32)

    rows = [0, 2, 3, 5, 6, 7]
    idx_mapping = torch.tensor(rows, device=device, dtype=torch.int32)

    prompt_bin_mask = torch.full((max_reqs, num_words), -1, dtype=torch.int32, device=device)
    output_bin_counts = torch.full((max_reqs, vocab_size), -1, dtype=torch.int32, device=device)

    _bincount_torch(
        idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        int(prefill_len.max()),
    )

    ref_mask, ref_counts = reference(
        rows, all_token_ids.cpu(), prompt_len.cpu(), prefill_len.cpu(), vocab_size, num_words
    )

    got_mask = prompt_bin_mask.cpu()[rows]
    got_counts = output_bin_counts.cpu()[rows]

    ok = True
    if not torch.equal(got_mask, ref_mask):
        bad = (got_mask != ref_mask).nonzero()
        print(f"FAIL: prompt_bin_mask differs at {bad.shape[0]} positions, first {bad[:3].tolist()}")
        ok = False
    else:
        print(f"prompt_bin_mask   matches reference  ({got_mask.shape[0]}x{got_mask.shape[1]} words)")

    if not torch.equal(got_counts, ref_counts):
        bad = (got_counts != ref_counts).nonzero()
        print(f"FAIL: output_bin_counts differs at {bad.shape[0]} positions, first {bad[:3].tolist()}")
        ok = False
    else:
        total = int(ref_counts.sum())
        print(f"output_bin_counts matches reference  ({total} counted tokens)")

    # Rows not selected must be untouched, since the caller relies on that.
    untouched = [r for r in range(max_reqs) if r not in rows]
    if not (prompt_bin_mask.cpu()[untouched] == -1).all():
        print("FAIL: unselected rows of prompt_bin_mask were modified")
        ok = False
    if not (output_bin_counts.cpu()[untouched] == -1).all():
        print("FAIL: unselected rows of output_bin_counts were modified")
        ok = False
    if ok:
        print("unselected rows untouched")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
