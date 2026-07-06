#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scGPT Fine-Tuning Script
========================
Fine-tune scGPT pre-trained model on reference scRNA-seq data for
generating gene expression embeddings used by CarHE.

Dependencies:
    pip install scgpt

scGPT Pre-trained Checkpoint:
    Download the 'pan_cancer' checkpoint from:
    https://github.com/bowang-lab/scGPT

    Place it at: scgpt_checkpoints/scgpt/pan_cancer/
    Expected files: best_model.pt, args.json, vocab.json

Usage:
    python scgpt_finetune.py --adata reference_data.h5ad --checkpoint ./scgpt_checkpoints/scgpt/pan_cancer
"""

import sys, os, copy, gc, json, time, argparse, warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import scanpy as sc
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed, load_pretrained

warnings.filterwarnings("ignore")


def prepare_data_loader(
    adata, vocab, config, special_tokens, mask_ratio, mask_value, pad_value,
    pad_token, max_seq_len, per_seq_batch_sample=True
):
    """Tokenize and create DataLoaders from AnnData"""
    # Preprocess
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=10,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=False,
        subset_hvg=config.n_hvg,
        hvg_flavor="seurat",
        binning=config.n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="batch_id")

    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].A
        if hasattr(adata.layers[input_layer_key], 'A')
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()
    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)
    num_batch_types = len(set(batch_ids))

    # Split
    train_data, valid_data, train_batch, valid_batch = train_test_split(
        all_counts, batch_ids, test_size=0.1, shuffle=True
    )

    # Tokenize
    gene_ids = np.array(vocab(genes), dtype=int)
    tokenized_train = tokenize_and_pad_batch(
        train_data, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value,
        append_cls=True, include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data, gene_ids, max_len=max_seq_len, vocab=vocab,
        pad_token=pad_token, pad_value=pad_value,
        append_cls=True, include_zero_gene=True,
    )

    print(f"Train samples: {tokenized_train['genes'].shape[0]}")
    print(f"Valid samples: {tokenized_valid['genes'].shape[0]}")

    return tokenized_train, tokenized_valid, train_batch, valid_batch, num_batch_types


def train_epoch(model, loader, criterion, optimizer, scaler, config,
                vocab, mask_value, pad_value, mask_ratio, pad_token):
    """Train for one epoch"""
    model.train()
    total_loss, total_error, num_batches = 0.0, 0.0, len(loader)

    for batch_data in loader:
        input_gene_ids = batch_data["gene_ids"].to(config.device)
        input_values = batch_data["values"].to(config.device)
        target_values = batch_data["target_values"].to(config.device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                MVC=True, ECS=config.ecs_thres > 0,
            )
            masked_positions = input_values.eq(mask_value)
            loss = criterion(output_dict["mlm_output"], target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=not scaler.is_enabled())
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )
        total_loss += loss.item()
        total_error += mre.item()

    return total_loss / num_batches, total_error / num_batches


def main():
    parser = argparse.ArgumentParser(description="scGPT Fine-Tuning")
    parser.add_argument("--adata", required=True, help="Reference AnnData path")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to scGPT pre-trained model dir (contains best_model.pt)")
    parser.add_argument("--output_dir", default="./scgpt_finetuned",
                        help="Output dir for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    class Config: pass
    config = Config()
    config.load_model = args.checkpoint
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.device = device
    config.n_hvg = 1200
    config.n_bins = 51
    config.mask_ratio = 0.4
    config.schedule_ratio = 0.9
    config.dropout = 0.2
    config.ecs_thres = 0.8
    config.amp = (device.type == "cuda")
    config.save_eval_interval = 5

    set_seed(42)

    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    pad_token, mask_value, pad_value = "<pad>", -1, -2
    max_seq_len = config.n_hvg + 1

    # Load pre-trained model
    model_dir = Path(config.load_model)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab: vocab.append_token(s)

    with open(model_dir / "args.json") as f:
        model_configs = json.load(f)

    ntokens = len(vocab)
    model = TransformerModel(
        ntokens, model_configs["embsize"], model_configs["nheads"],
        model_configs["d_hid"], model_configs["nlayers"],
        vocab=vocab, dropout=config.dropout,
        pad_token=pad_token, pad_value=pad_value,
        do_mvc=True, do_dab=True, use_batch_labels=False,
        num_batch_labels=2, domain_spec_batchnorm=False,
        n_input_bins=config.n_bins, ecs_threshold=config.ecs_thres,
        explicit_zero_prob=True, use_fast_transformer=True, pre_norm=False,
    )
    load_pretrained(model, torch.load(model_dir / "best_model.pt", map_location=device), verbose=False)
    model.to(device)

    # Freeze backbone, train only attention + decoders
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.transformer_encoder.layers:
        for p in layer.self_attn.parameters():
            p.requires_grad = True
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in
               ['decoder', 'cls_decoder', 'mvc_decoder', 'grad_reverse_discriminator']):
            p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  Trainable: {trainable:,}")

    # Load data
    print(f"Loading {args.adata}...")
    adata = sc.read_h5ad(args.adata)
    adata.obs['batch_id'] = adata.obs['sample_id'].astype("category").cat.codes.values
    adata.var["gene_name"] = adata.var.index.tolist()

    # Match genes to vocab
    adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"]]
    matched = (np.array(adata.var["id_in_vocab"]) >= 0).sum()
    print(f"Genes matched in vocab: {matched}/{len(adata.var)}")
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # Prepare data
    tokenized_train, tokenized_valid, train_batch, valid_batch, num_batch_types = \
        prepare_data_loader(adata, vocab, config, special_tokens,
                          config.mask_ratio, mask_value, pad_value,
                          pad_token, max_seq_len)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    criterion = masked_mse_loss

    # Train
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        # Re-randomize masking each epoch
        masked_train = random_mask_value(
            tokenized_train["values"], mask_ratio=config.mask_ratio,
            mask_value=mask_value, pad_value=pad_value
        )
        masked_valid = random_mask_value(
            tokenized_valid["values"], mask_ratio=config.mask_ratio,
            mask_value=mask_value, pad_value=pad_value
        )
        train_pt = {"gene_ids": tokenized_train["genes"], "values": masked_train,
                     "target_values": tokenized_train["values"],
                     "batch_labels": torch.from_numpy(train_batch).long()}
        valid_pt = {"gene_ids": tokenized_valid["genes"], "values": masked_valid,
                     "target_values": tokenized_valid["values"],
                     "batch_labels": torch.from_numpy(valid_batch).long()}

        train_ds = torch.utils.data.TensorDataset(
            train_pt["gene_ids"], train_pt["values"],
            train_pt["target_values"], train_pt["batch_labels"]
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

        train_loss, train_error = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            config, vocab, mask_value, pad_value, config.mask_ratio, pad_token
        )
        print(f"[Epoch {epoch}/{config.epochs}] loss={train_loss:.4f}  mre={train_error:.4f}")
        scheduler.step()

    # Save
    model_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Fine-tuned model saved to {model_path}")


if __name__ == "__main__":
    main()
