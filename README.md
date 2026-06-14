# SarvamVision

Document understanding system for Indian invoices — fine-tuning vision-language models to extract structured fields directly from document images.

## What We Are Doing

Indian businesses generate millions of invoices, receipts, and GST documents. Manually pulling out fields like totals, tax amounts, and vendor names is slow and error-prone. SarvamVision automates this: give it a document image, get back structured JSON — no OCR pipeline, no template matching required.

We fine-tune two vision-language models on Indian invoice data and benchmark them head-to-head:

- **Qwen2.5-VL-3B** via QLoRA (4-bit quantization, runs on single T4 GPU)
- **Florence-2-base** via full fine-tuning (230M params, fast inference)

**Extracted fields:** `vendor`, `date`, `total`, `gst`, `company`, `address`

## Results

| Model | F1 (macro) | Latency p95 | VRAM | Training |
|-------|-----------|-------------|------|----------|
| Qwen2.5-VL-3B (QLoRA) | **97.1%** | ~7500ms | ~9 GB | 3 epochs, 537 steps |
| Florence-2-base (fine-tuned) | **100.0%** | 221ms | ~0.9 GB | 5 epochs |
| Florence-2-base (baseline) | 0.0% | 181ms | ~0.5 GB | — |

Evaluated on 10-sample validation set. Kaggle free T4 GPU (16 GB VRAM).
Qwen loss: 16 → 0.1 over 3 epochs. Florence-2 loss: 4.0 → 0.10 over 5 epochs.

## Project Structure

```
SarvamVision/
├── data/
│   ├── raw/                    # Original invoice images
│   └── formatted/              # JSONL conversation format
│       ├── train.jsonl
│       └── validation.jsonl
├── training/
│   ├── train_qwen.py           # QLoRA fine-tuning — Qwen2.5-VL-3B
│   ├── train_florence.py       # Full fine-tuning — Florence-2
│   └── configs/
│       └── qwen_qlora.yaml
├── evaluation/
│   ├── evaluate.py             # Field F1, latency, hallucination metrics
│   └── benchmark_report.py     # Side-by-side comparison table
├── checkpoints/
│   ├── qwen2.5-vl-3b-qlora/final/
│   └── florence-2-base-ft/final/
├── reports/
│   └── benchmark.md            # Auto-generated results
├── inference/
│   ├── optimize.py
│   └── batching.py
├── rag/
│   ├── pipeline.py
│   └── visual_search.py
├── deployment/
│   └── api.py
└── frontend/
    └── app.py
```

## Training

```bash
# Qwen2.5-VL-3B — QLoRA 4-bit
python training/train_qwen.py

# Florence-2 — full fine-tune
python training/train_florence.py
```

## Evaluation

```bash
python evaluation/benchmark_report.py --max-samples 100 --latency-runs 10
```

Results saved to `reports/benchmark.md`.

## Key Design Decisions

**QLoRA for Qwen** — 3B model needs ~24 GB VRAM in full precision. 4-bit NF4 quantization + LoRA adapters (r=16) bring it down to ~9 GB with only 20M trainable parameters.

**Full fine-tune for Florence-2** — At 230M params it fits comfortably on T4 without quantization, so no LoRA needed.

**`<CAPTION>` task token for Florence-2** — Florence-2 has no native key-value extraction task. We repurpose the caption task and train it to output JSON instead of free-form text.

**Field-level F1 metric** — Each invoice field is evaluated independently. Fields with no ground truth values are excluded from the macro average so empty fields don't inflate scores.

## Tech Stack

- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base)
- [PEFT / LoRA](https://github.com/huggingface/peft) — parameter-efficient fine-tuning
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [TRL SFTTrainer](https://github.com/huggingface/trl) — supervised fine-tuning
- PyTorch + HuggingFace Transformers
- Kaggle T4 GPU (free tier)
