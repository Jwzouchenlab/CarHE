# -*- coding: utf-8 -*-
"""CarHE Training Script
统一使用 AnnData (h5ad) 格式。所有数据集必须先转换为 h5ad。

Usage:
    python train.py --adata ../data/xenium_prostate.h5ad --epochs 100
    python train.py --adata ../data/brca_data.h5ad --batch_size 64 --lr 0.0005
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import logging
import scanpy as sc

from model import HIPT_CLIP_Model
from get_adata import build_loaders_adata
from utils import AvgMeter, get_lr
from config import CFG


def setup_logger(log_file, level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def check_batch(batch, device, logger):
    return {k: v.to(device) if k != 'barcode' else v for k, v in batch.items()}


def train_epoch(model, train_loader, optimizer, device, logger, epoch):
    loss_meter = AvgMeter()
    model.train()
    with tqdm(train_loader, total=len(train_loader), unit="batch", leave=False) as tepoch:
        for i, batch in enumerate(tepoch):
            try:
                batch = check_batch(batch, device, logger)
                loss = model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                tepoch.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA out of memory: {e}")
                    torch.cuda.empty_cache()
                    return None
                else:
                    logger.exception(f"An unexpected error occurred during training: {e}")
                    return None
    return loss_meter


def test_epoch(model, test_loader, device, logger, epoch):
    loss_meter = AvgMeter()
    model.eval()
    with torch.no_grad(), tqdm(test_loader, total=len(test_loader), unit="batch", leave=False) as tepoch:
        for i, batch in enumerate(tepoch):
            try:
                batch = check_batch(batch, device, logger)
                loss = model(batch)
                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                tepoch.set_postfix(valid_loss=loss_meter.avg)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA out of memory: {e}")
                    torch.cuda.empty_cache()
                    return None
                else:
                    logger.exception(f"An unexpected error occurred during validation: {e}")
                    return None
    return loss_meter


def main(args):
    logger = setup_logger(args.log_file, level=logging.INFO)
    logger.info("Starting CarHE training...")
    logger.info(f"Arguments: {args}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = HIPT_CLIP_Model().to(device)
    logger.info('Model initialized.')

    logger.info(f'Loading data from {args.adata}...')
    train_loader, test_loader = build_loaders_adata(
        adata_path=args.adata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        logger.info(f"Epoch: {epoch + 1}/{args.epochs}")
        try:
            train_loss = train_epoch(model, train_loader, optimizer, device, logger, epoch)
            if train_loss is None:
                break

            with torch.no_grad():
                test_loss = test_epoch(model, test_loader, device, logger, epoch)
                if test_loss is None:
                    break

            logger.info(f"Epoch:{epoch + 1}, Train Loss: {train_loss.avg:.6f}, Val Loss: {test_loss.avg:.6f}")

            if test_loss.avg < best_val_loss:
                best_val_loss = test_loss.avg
                best_epoch = epoch + 1
                os.makedirs(args.exp_name, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.exp_name, "best_model.pt"))
                logger.info(f"Saved Best Model! Val Loss: {best_val_loss:.6f}")
        except Exception as e:
            logger.exception(f"An error occurred during training: {e}")
            break

    os.makedirs(args.exp_name, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.exp_name, "model.pt"))
    logger.info(f"Training completed! Best Val Loss: {best_val_loss:.6f} at Epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CarHE: H&E to Spatial Transcriptomics Training")

    parser.add_argument("--adata", type=str, default=CFG.default_adata_path,
                        help="AnnData h5ad 文件路径")
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)
    parser.add_argument("--exp_name", type=str, default=os.path.join(CFG.checkpoint_dir, "train"))
    parser.add_argument("--log_file", type=str, default=os.path.join(CFG.log_dir, "train_log.txt"))
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    main(args)
