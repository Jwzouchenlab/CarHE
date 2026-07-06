import numpy as np
import glob
import natsort
import tifffile
import os
import cv2
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import argparse
import sys


def process_nucleus(nuc_id):
    nuc_coords = np.where(seg_hist == nuc_id)
    cells = seg_xenium[nuc_coords[0], nuc_coords[1]]
    cells_unique, counts = np.unique(cells, return_counts=True)
    nonzero = np.nonzero(cells_unique)[0]
    matches = list(cells_unique[nonzero])
    nonzero_counts = counts[cells_unique != 0]
    n_overlap = len(nonzero)
    if n_overlap > 0:
        orig_size = [len(nuc_coords[0])] * n_overlap
        orig_id = [nuc_id] * n_overlap
        return orig_id, matches, nonzero_counts, orig_size
    else:
        return [], [], [], []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp_seg_hist",
        default="he_image_nuclei_seg_microns.tif",
        type=str,
        help="file path to nuclei segmented from H&E image",
    )
    parser.add_argument(
        "--fp_seg_xenium",
        default="xenium_nuclei.tif",
        type=str,
        help="file path to nuclei segmented from DAPI image",
    )
    parser.add_argument(
        "--fp_cgm",
        default="cell_gene_matrix.csv",
        type=str,
        help="file path to Xenium cell gene matrix from 4_get_xenium_cell_gene_matrix.py",
    )
    parser.add_argument(
        "--fp_out_matched_nuclei",
        default="matched_nuclei.csv",
        type=str,
        help="file path of output csv",
    )
    parser.add_argument(
        "--n_processes",
        default=24,
        type=int,
        help="max number of cpus to use",
    )
    parser.add_argument(
        "--min_overlap_fraction",
        default=0.5,
        type=float,
        help="min overlap fraction of H&E nucleus area between 2 nuclei",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)

    config = parser.parse_args()

    dir_output = config.dir_output
    if not os.path.exists(dir_output):
        sys.exit(f"Data not found in --dir_output: {dir_output}")

    fp_seg_xenium = os.path.join(config.dir_output, config.fp_seg_xenium)
    seg_xenium = tifffile.imread(fp_seg_xenium)

    fp_seg_hist = os.path.join(config.dir_output, config.fp_seg_hist)
    seg_hist = tifffile.imread(fp_seg_hist)

    # seg_hist = cv2.resize(
    #     seg_hist.astype(np.float32),
    #     (seg_xenium.shape[1], seg_xenium.shape[0]),
    #     interpolation=cv2.INTER_NEAREST,
    # )
    # seg_hist = seg_hist.astype(np.uint32)
    if seg_xenium.shape != seg_hist.shape:
        sys.exit(
            "2 segmentations need to be the same shape, recommended resolution for speed, 1 pixel = 1 micron"
        )

    nuc_unique_ids = np.unique(seg_hist)

    # Determine the number of CPUs to use, defaulting to all available CPUs
    num_cpus = min(
        cpu_count() - 2, config.n_processes
    )  # Set the maximum number of CPUs here
    print("Processing using CPUs:", num_cpus)

    # Multiprocessing
    with Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(process_nucleus, nuc_unique_ids),
                total=len(nuc_unique_ids),
            )
        )

    original_ids = []
    matching_ids = []
    overlap_amount = []
    original_size = []

    for orig_id, match_id, overlap_count, orig_size in results:
        original_ids.extend(orig_id)
        matching_ids.extend(match_id)
        overlap_amount.extend(overlap_count)
        original_size.extend(orig_size)

    original_ids = np.array(original_ids)
    matching_ids = np.array(matching_ids)
    overlap_amount = np.array(overlap_amount)
    original_size = np.array(original_size)

    # print(
    #     original_ids.shape,
    #     matching_ids.shape,
    #     overlap_amount.shape,
    #     original_size.shape,
    # )

    combined = np.vstack((original_ids, matching_ids, overlap_amount, original_size)).T
    df = pd.DataFrame(
        combined, columns=["id_histology", "id_xenium", "overlap", "size_pix_histology"]
    )

    # Group by 'id_histology' and find the index with the largest 'overlap' for each group
    max_overlap_indices = df.groupby("id_histology")["overlap"].idxmax()

    # Use the indices to get the corresponding 'id_xenium' values
    max_match_values = df.loc[max_overlap_indices, "id_xenium"].tolist()
    orig_values = df.loc[max_overlap_indices, "id_histology"].tolist()
    overlap_sizes = df.loc[max_overlap_indices, "overlap"].tolist()
    orig_sizes = df.loc[max_overlap_indices, "size_pix_histology"].tolist()

    # Create a DataFrame from the lists
    df_match = pd.DataFrame(
        {
            "id_histology": orig_values,
            "id_xenium": max_match_values,
            "overlap": overlap_sizes,
            "size_pix_histology": orig_sizes,
        }
    )

    # remove background
    df_match = df_match[df_match["id_histology"] != 0]

    # Save the DataFrame to a CSV file
    fp_out_matched_nuclei = os.path.join(dir_output, config.fp_out_matched_nuclei)
    df_match.to_csv(fp_out_matched_nuclei, index=False)

    # Filter nuclei

    # for each match, keep one id_histology that has the largest overlap and also overlap > threshold
    # > 50% overlap of the nuclei size in the H&E image
    df_match = df_match[
        df_match["overlap"]
        >= config.min_overlap_fraction * df_match["size_pix_histology"]
    ]

    df_match = df_match.sort_values(
        by=["id_xenium", "overlap"], ascending=[True, False]
    )

    # Drop duplicates based on 'match', keeping only the first occurrence
    df_match = df_match.drop_duplicates(subset="id_xenium", keep="first")

    mapping_dict = dict(zip(df_match["id_xenium"], df_match["id_histology"]))

    df_match_sorted = df_match.sort_values(by="id_histology")

    fp_out_matched_nuclei_f = fp_out_matched_nuclei.replace('.csv', '_filtered.csv')
    df_match_sorted.to_csv(fp_out_matched_nuclei_f, index=False)

    # get expression matrix and filter cells
    fp_cgm = os.path.join(dir_output, config.fp_cgm)
    df_expr = pd.read_csv(fp_cgm, index_col=0)
    if 'cell_id' not in df_expr.columns:
        df_expr['cell_id'] = df_expr.index

    # IDs not in the dict will be nan
    df_expr["cell_id"] = df_expr["cell_id"].map(mapping_dict)
    df_expr.set_index("cell_id", inplace=True, drop=False)
    df_expr.index.name = None
    df_expr.dropna(inplace=True)
    df_expr.drop('cell_id', axis=1, inplace=True)
    df_expr.index = df_expr.index.astype(int)

    fp_out_expr_mat = fp_cgm.replace('.csv', '_filtered.csv')
    df_expr.to_csv(fp_out_expr_mat)
    print("Saved", fp_out_expr_mat)
