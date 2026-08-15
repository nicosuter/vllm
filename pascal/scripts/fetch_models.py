"""Download the models under test into the PVC.

Kept as a file rather than an inline heredoc: quoting a nested f-string through
kubectl exec into bash is exactly the kind of thing that silently mangles.
"""

import sys

from huggingface_hub import snapshot_download

# Quantized variants are preferred wherever the headline model ships
# unquantized, because 8 GB does not leave room for bf16 weights plus a KV
# cache. Both substitutions below are compressed-tensors W4A16 — the same format
# the gate model uses, so they land on the Triton W4A16 kernel we have already
# verified numerically on sm_61.
REPOS = [
    # 2.34 GB W4A16, replacing ibm-granite/granite-4.1-3b (6.82 GB bf16).
    "cyankiwi/granite-4.1-3b-AWQ-INT4",
    # 4.26 GB bf16 -> fp16 on load; fits without quantizing.
    "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3-VL-Reranker-2B",
    # 8.32 GB W4A16, replacing google/gemma-4-E2B-it-qat-q4_0-unquantized
    # (10.2 GB bf16). Still over 8 GB because Gemma 4 E2B's per-layer
    # embeddings stay unquantized in every variant, so this one needs
    # cpu_offload_gb.
    "google/gemma-4-E2B-it-qat-w4a16-ct",
]


def main() -> int:
    wanted = sys.argv[1:] or REPOS
    for repo in wanted:
        name = repo.split("/")[-1]
        target = f"/work/models/{name}"
        print(f"--> {repo} -> {target}", flush=True)
        try:
            snapshot_download(repo, local_dir=target)
            print(f"    ok {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
