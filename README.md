# DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning ✨

📄 NeurIPS 2025 · [Paper](https://arxiv.org/abs/2510.17489) · 🤗 [Model: heyongxin233/DETree](https://huggingface.co/heyongxin233/DETree) · 🧪 [Dataset: heyongxin233/RealBench](https://huggingface.co/datasets/heyongxin233/RealBench)

## Table of Contents
- [Introduction](#introduction)
- [RealBench Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Demo & Inference](#demo--inference)
- [Training](#training)
- [Scripts Overview](#scripts-overview)
- [Citation](#Citation)

## 🎯 Introduction

DETree is a novel representation learning-based detection framework designed to address the challenges of identifying human-AI hybrid text in complex real-world scenarios. We construct RealBench, a large-scale benchmark dataset that encompasses diverse modes of human–AI collaborative writing. By explicitly modeling the hierarchical relationships among text sources, DETree reveals that hybrid texts generated through human–AI collaboration exhibit stronger AI traces than human characteristics. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple benchmark tasks and maintains strong generalization capabilities under low-supervision conditions and severe distribution shifts. 

This repository contains the full training and inference stack used in our paper. Every stage is exposed as a standalone Python module or shell script so you can mix, match, and customise the pipeline without digging through hidden orchestration logic.

## 🗂️ RealBench Dataset

RealBench consists of human-written, machine-generated, and human–AI collaborative texts, constructed following the same settings as described in the paper. 
Download the dataset from the Hugging Face Hub and keep the folder structure intact:

```bash
huggingface-cli download --repo-type dataset heyongxin233/RealBench --local-dir /path/to/RealBench
```

The repository is organised by benchmark family (e.g., `Deepfake/`, `OUTFOX/`, `RAID/`, `TuringBench/`, `M4_monolingual/`, `M4_multilingual/`). Each family contains attack-specific folders such as `no_attack/`, `extend/`, `paraphrase_by_llm/`, `perplexity_attack/`, or `synonym/`, and every folder holds split-wise JSONL dumps (`train.jsonl`, `valid.jsonl`, `test.jsonl`, plus extras like `test_ood.jsonl` where available). A compact view of the default layout looks like:

```text
RealBench/
├── Deepfake/
│   ├── no_attack/
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   ├── paraphrase_by_llm/
│   │   └── ...
│   └── (extend | perplexity_attack | polish | synonym | translate)/
├── OUTFOX/
│   └── ...
├── RAID/ and RAID_extra/
│   └── ...
├── TuringBench/
│   └── ...
├── M4_monolingual/ and M4_multilingual/
│   └── ...
└── embbedings/
    ├── mage_center10k.pt
    └── priori1_center10k.pt
```

Every JSON record exposes the keys consumed by the training and evaluation scripts—`text` (content), `label` (binary human/AI tag), `src` (generator identity), and `id` (stable sample identifier). Downstream metrics treat label `'0'` as human and `'1'` as machine, matching the evaluation helpers. Some sub-benchmarks add optional metadata fields (topic, prompt, attack recipe, etc.); you can leave those untouched, and DETree will simply pass them through the pipeline.

Inside the root folder you will also find an `embbedings/` directory containing two ready-to-use databases:

- `embbedings/mage_center10k.pt` – embeddings built from the MAGE training split and compressed to 10k prototypes.
- `embbedings/priori1_center10k.pt` – embeddings covering the full RealBench AI/human data, compressed with the same hyper-parameters.

> 💡 **Plug-and-play embeddings:** Both files follow the exact schema produced by [`scripts/gen_emb.sh`](scripts/gen_emb.sh), making them drop-in replacements for generated checkpoints. Point the demo or inference commands below at either file for instant results—no retraining required.

## 🧠 Model

The finetuned detector checkpoint is published as [heyongxin233/DETree](https://huggingface.co/heyongxin233/DETree). Load it via the Hugging Face `AutoModel` APIs or point the scripts in this repository at the hub identifier. The model is compatible with the compressed RealBench embeddings out of the box, so you can evaluate or serve DETree immediately after downloading the assets above.

## ⚙️ Installation

1. Clone the repository and enter the folder.
   ```bash
   git clone https://github.com/heyongxin233/DETree.git
   cd DETree
   ```
2. Create a Python environment and install the runtime dependencies (PyTorch—2.8.0 recommended—Transformers, the latest Lightning release, GPU-enabled FAISS, Gradio, etc.).
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > 🔧 **Version highlights:** `requirements.txt` keeps `torch` flexible (2.8.0 is recommended), leaves `lightning` unpinned so you always receive the newest release, and installs `faiss-gpu` without a version constraint for compatibility with your CUDA setup.
3. Add the repository to your `PYTHONPATH` so the CLI modules resolve when executed from the project root.
   ```bash
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   ```

## 🚀 Demo & Inference

### 💻 Command-line demo

Run the lightweight CLI demo with either of the published embedding databases:

```bash
python example/infer.py \
  --database-path /path/to/RealBench/embbedings/priori1_center10k.pt \
  --model-name-or-path heyongxin233/DETree \
  --text "Large language models are changing the world."
```

The script accepts repeated `--text` arguments or an input file/JSONL via `--input-file` and prints the DETree label together with calibrated human/AI probabilities.

### 🌐 Gradio web UI

Launch the interactive interface for live demos and sharing:

```bash
python example/web_demo.py \
  --database-path /path/to/RealBench/embbedings/mage_center10k.pt \
  --model-name-or-path heyongxin233/DETree \
  --host 0.0.0.0 --port 7860
```

Both demo entry points automatically expose controls for switching the embedding layer, changing the kNN neighbourhood size, and adjusting the detection threshold.

## 🏋️ Training

### Tree-Structured Contrastive Learning

Align the encoder with the HAT using a tree-structured contrastive loss. We provide prebuilt trees in `HAT_structure/`; using the corresponding tree directly can reproduce the results in the paper.

Run [`scripts/train_detree.sh`](scripts/train_detree.sh): train DETree on the build HAT.

### Reproducing the Paper

The training workflow mirrors the two-stage procedure described in the paper:

1. **Stage 1: Supervision Contrastive Learning**
   - [`scripts/extract_pcl_tree.sh`](scripts/extract_pcl_tree.sh): build the handcrafted tree from the RealBench JSONL files.
   - [`scripts/train_detree.sh`](scripts/train_detree.sh): train DETree with LoRA adapters on the handcrafted tree.
   - [`scripts/gen_emb.sh`](scripts/gen_emb.sh): export the Stage 1 embedding database.
2. **Stage 2:Tree-Structured Contrastive Learning**
   - [`scripts/build_hat_tree.sh`](scripts/build_hat_tree.sh): compute similarity matrices from the Stage 1 embeddings and derive the hierarchical tree (per encoder layer).
   - [`scripts/train_detree.sh`](scripts/train_detree.sh): train DETree on the new tree.
3. **Evaluation and deployment**
   - [`scripts/merge_lora.sh`](scripts/merge_lora.sh): merge the LoRA adapter into the base checkpoint for standalone inference.
   - [`scripts/compress_database.sh`](scripts/compress_database.sh): cluster embeddings into compact and balanced prototypes.
   - [`scripts/test_database_score_knn.sh`](scripts/test_database_score_knn.sh) or [`scripts/test_score_knn.sh`](scripts/test_score_knn.sh): evaluate the merged model either against a cached database or directly on JSONL corpora.

Each script lists all configurable arguments at the top—edit the path and hyper-parameter variables, then run the file directly.

## 📜 Scripts Overview

| Script | Purpose |
| --- | --- |
| [`scripts/extract_pcl_tree.sh`](scripts/extract_pcl_tree.sh) | Generate the SCL tree used during stage 1 training. |
| [`scripts/train_detree.sh`](scripts/train_detree.sh) | Launch DETree training with configurable optimisation, LoRA, and data options. |
| [`scripts/gen_emb.sh`](scripts/gen_emb.sh) | Export embedding databases from a trained checkpoint. |
| [`scripts/build_hat_tree.sh`](scripts/build_hat_tree.sh) | Create similarity matrices and HAT from embeddings. |
| [`scripts/merge_lora.sh`](scripts/merge_lora.sh) | Merge a LoRA adapter into the base RoBERTa model. |
| [`scripts/compress_database.sh`](scripts/compress_database.sh) | Cluster embeddings into compact prototypes for efficient inference. |
| [`scripts/test_database_score_knn.sh`](scripts/test_database_score_knn.sh) | Evaluate checkpoints against a saved embedding database. |
| [`scripts/test_score_knn.sh`](scripts/test_score_knn.sh) | Evaluate checkpoints directly on JSONL corpora without a cached database. |

## 📚 Citation

If you use our code or findings in your research, please cite us as:
```
@article{nips2025detree,
  title={DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning},
  author={He, Yongxin and Zhang, Shan and Cao, Yixuan and Ma, Lei and Luo, Ping},
  journal={arXiv preprint arXiv:2510.17489},
  year={2025}
}
```

