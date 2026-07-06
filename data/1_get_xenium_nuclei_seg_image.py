import numpy as np
import tifffile
import pandas as pd
import cv2
from tqdm import tqdm
import multiprocessing
import argparse
import sys
import os
import glob
import natsort
import multiprocessing as mp
import random


def process_patch(ws, we, df, cell_ids, max_w, max_h, dir_output):
    img_out = np.zeros((max_h, we - ws), dtype=np.uint32)

    df_chunk = df[(df["vertex_x"] >= ws) & (df["vertex_x"] < we)]
    cells_chunk = list(set(df_chunk["cell_id"].tolist()))
    # print(len(cells_chunk))

    cells_intersect = list(set(cell_ids) & set(cells_chunk))

    df_cells = df[df["cell_id"].isin(cells_intersect)]

    # print(ws, we, df_cells.shape)

    for cell in tqdm(cells_intersect, total=len(cells_intersect)):
        df_cell = df_cells[(df_cells["cell_id"] == cell)]
        if (df_cell["vertex_x"] >= we).any() or (df_cell["vertex_x"] < ws).any():
            continue

        df_cell = df_cell.assign(vertex_x=df_cell["vertex_x"] - ws)
        df_cell.loc[df_cell["vertex_x"] >= we, "vertex_x"] = we - 1
        df_cell.loc[df_cell["vertex_y"] >= max_h, "vertex_y"] = max_h - 1

        coords = [list(x) for x in zip(df_cell["vertex_x"], df_cell["vertex_y"])]
        polygon = np.array([coords], dtype=np.int32)
        mask = np.zeros((max_h, we - ws), dtype=np.uint8)
        mask = cv2.fillPoly(mask, polygon, 255)
        img_out = np.where(mask > 0, cell, img_out)

    tifffile.imwrite(
        f"{dir_output}/xenium_nuclei_{ws}.tif", img_out, photometric="minisblack"
    )


def get_n_processes(config_n_processes):
    if config_n_processes is None:
        n_processes = mp.cpu_count() - 2
    else:
        n_processes = (
            config_n_processes
            if config_n_processes <= mp.cpu_count()
            else mp.cpu_count()
        )
    return n_processes


def process_patch_wrapper(ws, we, df, cell_ids, max_w, max_h, dir_output):
    # Wrap the process_patch function to fit the Pool's expected signature
    process_patch(ws, we, df, cell_ids, max_w, max_h, dir_output)


def df_column_switch(df, column1, column2):
    cols = df.columns.tolist()
    i, j = cols.index(column1), cols.index(column2)
    cols[i], cols[j] = cols[j], cols[i]
    df.columns = cols
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp_boundaries",
        default="nucleus_boundaries.csv.gz",
        type=str,
        help="nuclei boundaries from Xenium file path",
    )
    parser.add_argument(
        "--fp_he_img",
        default="he_image.ome.tif",
        type=str,
        help="corresponding full resolution H&E image file path",
    )
    parser.add_argument(
        "--fp_out_nuclei_seg",
        default="xenium_nuclei.tif",
        type=str,
        help="output file path of Xenium nuclei segmentation",
    )
    parser.add_argument(
        "--fp_ids_out",
        default="cell_ids_xenium.csv",
        type=str,
        help="output file path of corresponding cell IDs in Xenium analysis files and segmentation image",
    )
    parser.add_argument(
        "--crop_fraction",
        default=0.2,
        type=float,
        help="process segmentation in patches - size as fraction of whole image",
    )
    parser.add_argument(
        "--overlap_fraction",
        default=0.1,
        type=float,
        help="process segmentation in patches - overlap fraction between crops",
    )
    parser.add_argument(
        "--n_processes",
        default=16,
        type=int,
        help="num cpus to use, or None for all cpus - 1",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)
    parser.add_argument(
        "--del_intm_files",
        default=True,
        type=bool,
        help="delete intermediate saved files",
    )

    config = parser.parse_args()

    dir_output = config.dir_output
    os.makedirs(dir_output, exist_ok=True)

    fp_csv = config.fp_boundaries
    df = pd.read_csv(fp_csv, index_col=None, compression="gzip")

    cell_ids_orig = df["cell_id"].unique().tolist()

    if all(isinstance(item, str) for item in cell_ids_orig):
        # map to integers if ct is string
        cell_ids = list(range(1, len(cell_ids_orig) + 1))
        df_cid = pd.DataFrame(cell_ids, index=cell_ids_orig, columns=["cell_id_num"])
        fp_out_cid = f"{dir_output}/xenium_cell_ids_dict.csv"
        df_cid.to_csv(fp_out_cid)

        df = df.merge(df_cid, how='left', left_on='cell_id', right_index=True)

        # make numerical IDs the default
        df = df_column_switch(df, "cell_id", "cell_id_num")
    else:
        cell_ids = cell_ids_orig.copy()

    # try on a small subset
    # cell_ids = random.sample(cell_ids, 200)

    he_img = tifffile.imread(config.fp_he_img)
    if he_img.shape[-1] == 3:
        max_h = he_img.shape[0]
        max_w = he_img.shape[1]
    elif he_img.shape[0] == 3:
        max_h = he_img.shape[1]
        max_w = he_img.shape[2]
    else:
        sys.exit("RGB channel not 0 or 2")

    smaller_side_size = min(max_h, max_w)
    every = int(round(config.crop_fraction * smaller_side_size))
    overlap = int(round(config.overlap_fraction * every))
    w_starts = list(np.arange(0, max_w - every, every - overlap))
    w_starts.append(max_w - every)

    #df = df.assign(vertex_x=df["vertex_x"] / 0.2125)
    #df = df.assign(vertex_y=df["vertex_y"] / 0.2125)
    df["vertex_x"] = df["vertex_x"].round(0).astype(int)
    df["vertex_y"] = df["vertex_y"].round(0).astype(int)

    # print(df["vertex_x"].min(), df["vertex_x"].max())
    # print(df["vertex_y"].min(), df["vertex_y"].max())

    n_processes = get_n_processes(config.n_processes)

    # Turn the vertices into segmentation masks
    # Process the whole image as crops

    # processes = []
    # for ws in w_starts:
    #     we = ws + every
    #     p = multiprocessing.Process(
    #         target=process_patch,
    #         args=(ws, we, df, cell_ids, max_w, max_h, dir_output),
    #     )
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.starmap(
            process_patch_wrapper,
            [
                (ws, ws + every, df, cell_ids, max_w, max_h, dir_output)
                for ws in w_starts
            ],
        )

    # combine the crops together to form whole image at full resolution
    print('Combining crops')

    fp_out = os.path.join(dir_output, config.fp_out_nuclei_seg)

    fp_sections = f"{dir_output}/xenium_nuclei_*.tif"
    fps = glob.glob(fp_sections)
    fps = natsort.natsorted(fps)
    n_sections = len(fps)

    final = np.zeros((max_h, max_w), dtype=np.uint32)

    for isec, fp in tqdm(enumerate(fps), total=n_sections):
        section = tifffile.imread(fp)

        fname = os.path.basename(fp).replace(".tif", "")

        ws = int(fname.split("_")[-1])
        we = ws + section.shape[1]

        final[:, ws:we] = np.where(section > 0, section, final[:, ws:we])

    resized_h = int(round(max_h * 0.2125))
    resized_w = int(round(max_w * 0.2125))
    final = cv2.resize(
        final.astype(np.float32),
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )
    final = final.astype(np.uint32)

    print("Total nuclei", len(np.unique(final)) - 1)

    tifffile.imwrite(fp_out, final, photometric="minisblack")

    if config.del_intm_files:
        print("Deleting intermediate files")
        # input("Press Enter to continue or CTRL+C to quit...")

        for fp in fps:
            os.remove(fp)
