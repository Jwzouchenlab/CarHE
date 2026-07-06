import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import gzip
import csv
import shutil
import os
from scipy.io import mmread
import argparse
import sys
import natsort


def ungz(fp):
    fp_out = fp.replace(".gz", "")
    if not os.path.exists(fp_out):
        with gzip.open(fp, "rb") as f_in:
            with open(fp_out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    return fp_out


def get_data(fp):
    print("Loading", fp)
    df = pd.read_csv(fp, sep="\t", index_col=False, header=None)
    df.reset_index(drop=True, inplace=True)
    # print(df.head())
    # print(df.shape)
    return df


def main(config):
    """
    Get cell-genes matrix using the provided cell feature matrix directory
        barcodes.tsv.gz
        features.tsv.gz
        matrix.mtx.gz
    Save as csv file
    """

    dir_output = config.dir_output
    os.makedirs(dir_output, exist_ok=True)

    # fp_genes = os.path.join(dir_output, config.fp_genes)
    # with open(fp_genes) as file:
    #     valid_genes = [line.rstrip() for line in file]

    dir_feature_matrix = config.dir_feature_matrix

    dir_feature_matrix_name, ext = os.path.splitext(dir_feature_matrix)
    if ext:
        if "tar.gz" in dir_feature_matrix:
            sys.exit("Please extract the files: tar -xvzf cell_feature_matrix.tar.gz")
        else:
            sys.exit("Please specify path to the cell_feature_matrix directory")

    fp_barcodes = os.path.join(dir_feature_matrix, config.fp_barcodes)
    fp_features = os.path.join(dir_feature_matrix, config.fp_features)
    fp_matrix = os.path.join(dir_feature_matrix, config.fp_matrix)

    # extract data
    fp_barcodes_intm = ungz(fp_barcodes)
    fp_features_intm = ungz(fp_features)
    fp_matrix_intm = ungz(fp_matrix)

    # ID of the cells
    barcodes = get_data(fp_barcodes_intm)

    if all(isinstance(item, str) for item in barcodes[0].unique().tolist()):
        map_cid = True
        fp_cid = f"{dir_output}/xenium_cell_ids_dict.csv"
        df_cid_dict = pd.read_csv(fp_cid, index_col=0)
        cid_dict = df_cid_dict['cell_id_num'].to_dict()
        # barcodes[0] = barcodes[0].map(cid_dict).astype(int)
    else:
        map_cid = False

    # print(barcodes.head())
    # exit()

    #    0
    # 0  1
    # 1  2
    # 2  3
    # 3  4
    # 4  5
    # (356746, 1)
    #             0
    # 0  aaaadpbp-1
    # 1  aaaaficg-1
    # 2  aaabbaka-1
    # 3  aaabbjoo-1
    # 4  aaablchg-1
    # (162254, 1)

    # names of the features
    features = get_data(fp_features_intm)
    feature_names = features.loc[features.iloc[:, 2] == "Gene Expression", 1]
    feature_names = feature_names.tolist()

    # feature by cell
    matrix = mmread(fp_matrix_intm)
    matrix_array = matrix.toarray()

    # filter gene names, sort, and save list
    valid_genes = feature_names.copy()
    transcripts_to_filter = [
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword_",
        "BLANK_",
        "Blank-",
        "NegPrb",
        "Unassigned",
    ]
    valid_genes = [
        item for item in valid_genes
        if not any(substr in item for substr in transcripts_to_filter)
    ]
    valid_genes = natsort.natsorted(valid_genes)
    fp_out_genes = os.path.join(dir_output, config.fp_genes)
    with open(fp_out_genes, "w") as f:
        for line in valid_genes:
            f.write(f"{line}\n")

    # Order the feature names (that matches the other cell key gene matrices)
    idx_order = [feature_names.index(x) for x in valid_genes]
    matrix_filtered = matrix_array[idx_order, :]

    # cell x gene
    matrix_filtered = matrix_filtered.T
    # print(matrix_filtered.shape)

    fp_out_matrix = os.path.join(dir_output, config.fp_out_matrix)

    df_matrix_filtered = pd.DataFrame(
        matrix_filtered, index=barcodes[0], columns=valid_genes
    )

    if map_cid:
        cid_dict = df_cid_dict["cell_id_num"].to_dict()
        df_matrix_filtered = df_matrix_filtered[
            df_matrix_filtered.index.isin(cid_dict.keys())
        ]
        df_matrix_filtered.index = df_matrix_filtered.index.map(cid_dict)

    df_matrix_filtered.to_csv(fp_out_matrix)

    print("Saved", fp_out_matrix)

    if config.del_intm_files:
        print("Deleting intermediate files")
        # input("Press Enter to continue or CTRL+C to quit...")

        os.remove(fp_barcodes_intm)
        os.remove(fp_features_intm)
        os.remove(fp_matrix_intm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir_feature_matrix",
        default="cell_feature_matrix",
        type=str,
        help="Xenium cell feature matrix dir path",
    )
    parser.add_argument(
        "--fp_barcodes",
        default="barcodes.tsv.gz",
        type=str,
        help="barcodes file name",
    )
    parser.add_argument(
        "--fp_features",
        default="features.tsv.gz",
        type=str,
        help="features file name",
    )
    parser.add_argument(
        "--fp_matrix",
        default="matrix.mtx.gz",
        type=str,
        help="matrix file name",
    )
    parser.add_argument(
        "--fp_genes",
        default="genes.txt",
        type=str,
        help="genes list file name from 1_get_gene_panel.py",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)
    parser.add_argument("--fp_out_matrix", default="cell_gene_matrix.csv", type=str)
    parser.add_argument(
        "--del_intm_files",
        default=True,
        type=bool,
        help="delete intermediate saved files",
    )
    config = parser.parse_args()
    main(config)
