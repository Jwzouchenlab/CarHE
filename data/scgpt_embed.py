#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scGPT Embedding Extraction
==========================
Run a fine-tuned scGPT model on spatial transcriptomics data to generate
gene expression embeddings (stored in adata.obsm['X_scGPT']).

The resulting h5ad file can be directly used by CarHE training.

Dependencies:
    pip install scgpt

Pre-trained + Fine-tuned Checkpoints:
    1. scGPT pan_cancer:  https://github.com/bowang-lab/scGPT
       Place at: scgpt_checkpoints/scgpt/pan_cancer/

    2. Fine-tuned model:  output of scgpt_finetune.py
       Default:  ./scgpt_finetuned/best_model.pt

Usage:
    # Use pre-trained scGPT directly (no fine-tuning)
    python scgpt_embed.py --adata ../data/xenium_prostate.h5ad \
        --checkpoint ./scgpt_checkpoints/scgpt/pan_cancer

    # Use fine-tuned scGPT
    python scgpt_embed.py --adata ../data/xenium_prostate.h5ad \
        --checkpoint ./scgpt_checkpoints/scgpt/pan_cancer \
        --finetuned ./scgpt_finetuned/best_model.pt
"""

import sys, os, json, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, load_pretrained

warnings.filterwarnings("ignore")


def get_embeddings(
    adata,
    model,
    vocab,
    special_tokens,
    pad_token="<pad>",
    pad_value=-2,
    batch_size=64,
    device="cuda",
):
    """Extract cell embeddings from scGPT model"""
    model.eval()

    genes = adata.var["gene_name"].tolist() if "gene_name" in adata.var else adata.var_names.tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    n_cells = adata.n_obs
    emb_dim = model.encoder.embedding.embedding_dim
    embeddings = np.zeros((n_cells, emb_dim), dtype=np.float32)

    # Process in batches
    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        batch_adata = adata[i:end]

        if hasattr(batch_adata.X, 'toarray'):
            expr = batch_adata.X.toarray()
        else:
            expr = batch_adata.X
        expr = np.asarray(expr, dtype=np.float32)

        # Build input: gene_ids (shared) + values per cell
        batch_genes = np.tile(gene_ids[None, :], (end - i, 1))
        batch_values = expr[:, :len(gene_ids)]

        # Pad to max_seq_len (append CLS token)
        max_len = min(len(gene_ids) + 1, 1201)
        input_genes = np.zeros((end - i, max_len), dtype=int)
        input_values = np.full((end - i, max_len), pad_value, dtype=np.float32)
        input_genes[:, :len(gene_ids)] = batch_genes[:, :max_len - 1]
        input_values[:, :len(gene_ids)] = batch_values[:, :max_len - 1]
        input_genes[:, len(gene_ids)] = vocab["<cls>"]

        src_key_padding_mask = torch.tensor(input_genes).eq(vocab[pad_token]).to(device)

        with torch.no_grad():
            output = model(
                torch.tensor(input_genes).to(device),
                torch.tensor(input_values).to(device),
                src_key_padding_mask=src_key_padding_mask,
            )
            # Use CLS token embedding
            cls_emb = output["cell_emb"]
            embeddings[i:end] = cls_emb.cpu().numpy()

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {end}/{n_cells} cells")

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="scGPT Embedding Extraction")
    parser.add_argument("--adata", required=True, help="Spatial transcriptomics h5ad file")
    parser.add_argument("--checkpoint", required=True,
                        help="scGPT pre-trained model dir (contains args.json, vocab.json, best_model.pt)")
    parser.add_argument("--finetuned", default=None,
                        help="Fine-tuned model weights (optional)")
    parser.add_argument("--output", default=None,
                        help="Output h5ad path (default: overwrites input)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load scGPT
    model_dir = Path(args.checkpoint)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]

    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_dir / "args.json") as f:
        model_configs = json.load(f)

    ntokens = len(vocab)
    model = TransformerModel(
        ntokens, model_configs["embsize"], model_configs["nheads"],
        model_configs["d_hid"], model_configs["nlayers"],
        vocab=vocab, dropout=0.0, pad_token="<pad>", pad_value=-2,
        do_mvc=False, do_dab=False, use_batch_labels=False,
        num_batch_labels=0, domain_spec_batchnorm=False,
        n_input_bins=51, explicit_zero_prob=False,
        use_fast_transformer=True, pre_norm=False,
    )

    # Load weights
    if args.finetuned:
        print(f"Loading fine-tuned weights: {args.finetuned}")
        model.load_state_dict(torch.load(args.finetuned, map_location=device), strict=False)
    else:
        load_pretrained(model, torch.load(model_dir / "best_model.pt", map_location=device), verbose=False)

    model.to(device)

    # Load data
    print(f"Loading: {args.adata}")
    adata = sc.read_h5ad(args.adata)
    if "gene_name" not in adata.var:
        adata.var["gene_name"] = adata.var_names.tolist()

    # Match genes
    adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"]]
    matched = (np.array(adata.var["id_in_vocab"]) >= 0).sum()
    print(f"Genes matched: {matched}/{len(adata.var)}")
    adata_orig = adata.copy()
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

    # Extract embeddings
    print("Extracting scGPT embeddings...")
    embeddings = get_embeddings(adata, model, vocab, special_tokens,
                               device=device, batch_size=args.batch_size)

    # Store back
    adata_orig.obsm["X_scGPT"] = np.zeros((adata_orig.n_obs, embeddings.shape[1]), dtype=np.float32)
    matched_idx = np.where(np.array(adata_orig.var["id_in_vocab"]) >= 0)[0]
    adata_orig.obsm["X_scGPT"][:, :] = embeddings  # Note: shape mismatch if genes filtered
    adata_orig.uns["scgpt_checkpoint"] = str(model_dir)

    # Save
    output_path = args.output or args.adata
    adata_orig.write_h5ad(output_path)
    print(f"Saved: {output_path}")
    print(f"Embedding shape: {adata_orig.obsm['X_scGPT'].shape}")


if __name__ == "__main__":
    main()
