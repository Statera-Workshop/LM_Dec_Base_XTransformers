# LM_Dec_Base_XTransformers

XTransformers - Base family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: LM_2023_x-transformers-pytorch
- Upstream / license: vendored from RichardMinsooGo/LM_2023_x-transformers-pytorch (see LICENSE in vendor folder if present)

## Differences from the source ([Unified] protocol)

- the repository's own x-transformers copy, library defaults (absolute positions)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 20.21 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.879 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.474 / test 1.453 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.354 / test 1.344 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 2311 MB, inference 1460 MB
- Peak GPU memory at length 8192 (batch 1): train None, inference None
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.27 (keys removed 0.24); 4 pairs @ 2048: 0.26 (keys removed 0.29); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.06 (keys removed 0.06)
- LM-06 memory update `lm06`: @512: latest 0.62, unchanged 0.03, stale 0.02; @2048: latest 0.48, unchanged 0.03, stale 0.03
- LM-06 memory update `lm06long`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check xt_repo
python run.py lm02 xt_repo      # enwik8 is downloaded to data/ on first use
```
