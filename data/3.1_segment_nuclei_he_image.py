import numpy as np
import tifffile
import os
from PIL import Image
import argparse
import sys
import glob
import natsort
import scipy.io as sio
import re
import shutil
import subprocess
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

try:
    import cv2
except Exception as e:
    cv2 = None
    print("WARNING: cv2 import failed:", repr(e))

try:
    import openslide
except Exception as e:
    openslide = None
    print("WARNING: openslide import failed:", repr(e))

try:
    import pyvips
except Exception as e:
    pyvips = None
    print("WARNING: pyvips import failed:", repr(e))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def ensure_uint8_rgb(img):
    img = np.asarray(img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]

    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
        if img.shape[-1] == 4:
            img = img[:, :, :3]

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape H,W,3, got shape={img.shape}")

    if img.dtype == np.uint8:
        out = img
    else:
        img_float = img.astype(np.float32)

        finite = np.isfinite(img_float)
        if not np.any(finite):
            out = np.zeros(img.shape, dtype=np.uint8)
        else:
            maxv = np.nanmax(img_float[finite])
            minv = np.nanmin(img_float[finite])

            if maxv <= 1.5 and minv >= 0:
                img_float = img_float * 255.0
            elif maxv > 255:
                img_float = (img_float - minv) / max(maxv - minv, 1e-6) * 255.0

            out = np.clip(img_float, 0, 255).astype(np.uint8)

    out = np.ascontiguousarray(out)
    return out


def normalize_rgb_shape(img):
    return ensure_uint8_rgb(img)


def rgb_to_gray_and_sat(img):
    """
    Return gray uint8 and saturation uint8.
    Prefer cv2; fallback to pure numpy if cv2 has numpy ABI issue.
    """
    img = ensure_uint8_rgb(img)

    if cv2 is not None:
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sat = hsv[:, :, 1]
            return gray.astype(np.uint8), sat.astype(np.uint8)
        except Exception as e:
            print("WARNING: cv2.cvtColor failed, fallback to numpy.")
            print("cv2 error:", repr(e))

    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    gray = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)

    rgb = img.astype(np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)

    sat_float = np.zeros_like(maxc, dtype=np.float32)
    nonzero = maxc > 0
    sat_float[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]

    sat = np.clip(sat_float * 255.0, 0, 255).astype(np.uint8)
    return gray, sat


def is_mostly_background(patch, white_thresh=240, white_ratio=0.85, min_std=8):
    patch = ensure_uint8_rgb(patch)
    gray, _ = rgb_to_gray_and_sat(patch)

    ratio_white = np.mean(gray > white_thresh)
    patch_std = np.std(gray)

    return (ratio_white >= white_ratio) or (patch_std < min_std)


def can_open_with_openslide(fp):
    if openslide is None:
        return False
    try:
        slide = openslide.OpenSlide(str(fp))
        slide.close()
        return True
    except Exception:
        return False


def get_image_hw_and_layout(fp_he_img):
    fp_he_img = str(fp_he_img)

    if can_open_with_openslide(fp_he_img):
        slide = openslide.OpenSlide(fp_he_img)
        width, height = slide.dimensions
        slide.close()
        return height, width, "openslide"

    if pyvips is not None:
        try:
            img = pyvips.Image.new_from_file(fp_he_img, access="sequential")
            return img.height, img.width, "pyvips"
        except Exception as e:
            print("WARNING: pyvips get size failed:", repr(e))

    with tifffile.TiffFile(fp_he_img) as tif:
        series = tif.series[0]
        shape = series.shape

        if len(shape) == 2:
            return shape[0], shape[1], "GRAY"
        elif len(shape) == 3:
            if shape[-1] in (3, 4):
                return shape[0], shape[1], "HWC"
            elif shape[0] in (3, 4):
                return shape[1], shape[2], "CHW"
            else:
                raise ValueError(f"Unsupported 3D image shape: {shape}")
        else:
            raise ValueError(f"Unsupported image shape: {shape}")


def pyvips_to_numpy_rgb(vips_img):
    """
    Convert pyvips image to numpy RGB uint8.
    """
    if vips_img.bands == 1:
        vips_img = vips_img.colourspace("srgb")
    elif vips_img.bands >= 3:
        vips_img = vips_img[:3]
    else:
        raise ValueError(f"Unsupported pyvips bands: {vips_img.bands}")

    if vips_img.format != "uchar":
        vips_img = vips_img.cast("uchar")

    mem = vips_img.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8)
    arr = arr.reshape(vips_img.height, vips_img.width, vips_img.bands)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[:, :, :3]

    arr = np.array(arr, dtype=np.uint8, copy=True)
    arr = np.ascontiguousarray(arr)
    return arr


def load_thumbnail_for_mask_openslide(fp_he_img, thumb_max_size=4000):
    slide = openslide.OpenSlide(str(fp_he_img))

    width_whole, height_whole = slide.dimensions

    scale = min(
        thumb_max_size / height_whole,
        thumb_max_size / width_whole,
        1.0
    )

    thumb_h = max(1, int(round(height_whole * scale)))
    thumb_w = max(1, int(round(width_whole * scale)))

    print(f"Original image size: H={height_whole}, W={width_whole}")
    print(f"Loading thumbnail with OpenSlide: H={thumb_h}, W={thumb_w}")

    thumb = slide.get_thumbnail((thumb_w, thumb_h))
    thumb_rgb = np.array(thumb.convert("RGB"), dtype=np.uint8, copy=True)
    thumb_rgb = np.ascontiguousarray(thumb_rgb)

    slide.close()

    return thumb_rgb, height_whole, width_whole


def load_thumbnail_for_mask_pyvips(fp_he_img, thumb_max_size=4000):
    if pyvips is None:
        raise RuntimeError("pyvips is not available.")

    img = pyvips.Image.new_from_file(str(fp_he_img), access="sequential")

    height_whole = img.height
    width_whole = img.width

    scale = min(
        thumb_max_size / height_whole,
        thumb_max_size / width_whole,
        1.0
    )

    thumb_h = max(1, int(round(height_whole * scale)))
    thumb_w = max(1, int(round(width_whole * scale)))

    print(f"Original image size: H={height_whole}, W={width_whole}")
    print(f"Loading thumbnail with pyvips: H={thumb_h}, W={thumb_w}")

    thumb = img.resize(scale)
    thumb_rgb = pyvips_to_numpy_rgb(thumb)

    return thumb_rgb, height_whole, width_whole


def load_thumbnail_for_mask_tifffile(fp_he_img, thumb_max_size=4000):
    print("Loading thumbnail with tifffile fallback.")

    with tifffile.TiffFile(fp_he_img) as tif:
        arr = tif.series[0].asarray(out="memmap")

        if arr.ndim == 2:
            h, w = arr.shape
            layout = "GRAY"
        elif arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                h, w = arr.shape[:2]
                layout = "HWC"
            elif arr.shape[0] in (3, 4):
                h, w = arr.shape[1], arr.shape[2]
                layout = "CHW"
            else:
                raise ValueError(f"Unsupported image shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        scale = min(thumb_max_size / h, thumb_max_size / w, 1.0)
        th = max(1, int(round(h * scale)))
        tw = max(1, int(round(w * scale)))

        print(f"Original image size: H={h}, W={w}")
        print(f"Loading thumbnail with tifffile: H={th}, W={tw}")

        ys = np.linspace(0, h - 1, th).astype(np.int64)
        xs = np.linspace(0, w - 1, tw).astype(np.int64)

        if layout == "GRAY":
            thumb = arr[np.ix_(ys, xs)]
        elif layout == "HWC":
            thumb = arr[np.ix_(ys, xs)]
        else:
            thumb = arr[:, ys[:, None], xs]
            thumb = np.moveaxis(thumb, 0, -1)

        thumb = ensure_uint8_rgb(thumb)

    return thumb, h, w


def load_thumbnail_for_mask(fp_he_img, thumb_max_size=4000):
    fp_he_img = str(fp_he_img)

    if can_open_with_openslide(fp_he_img):
        try:
            return load_thumbnail_for_mask_openslide(fp_he_img, thumb_max_size)
        except Exception as e:
            print("WARNING: OpenSlide thumbnail failed:", repr(e))

    if pyvips is not None:
        try:
            print(f"OpenSlide unsupported, fallback to pyvips: {fp_he_img}")
            return load_thumbnail_for_mask_pyvips(fp_he_img, thumb_max_size)
        except Exception as e:
            print("WARNING: pyvips thumbnail failed:", repr(e))

    return load_thumbnail_for_mask_tifffile(fp_he_img, thumb_max_size)


def generate_tissue_mask(
    thumb_rgb,
    gray_thresh=235,
    sat_thresh=20,
    morph_kernel=5,
    min_tissue_area=64,
):
    thumb_rgb = ensure_uint8_rgb(thumb_rgb)

    print("Before color conversion in generate_tissue_mask:")
    print("type:", type(thumb_rgb))
    print("shape:", thumb_rgb.shape)
    print("dtype:", thumb_rgb.dtype)
    print("C_CONTIGUOUS:", thumb_rgb.flags["C_CONTIGUOUS"])
    print("WRITEABLE:", thumb_rgb.flags["WRITEABLE"])

    gray, sat = rgb_to_gray_and_sat(thumb_rgb)

    tissue_mask = ((gray < gray_thresh) & (sat > sat_thresh)).astype(np.uint8)

    if cv2 is not None and morph_kernel is not None and morph_kernel > 1:
        try:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        except Exception as e:
            print("WARNING: cv2 morphology failed, skip morphology.")
            print("cv2 morphology error:", repr(e))

    if cv2 is not None:
        try:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                tissue_mask.astype(np.uint8),
                connectivity=8
            )

            cleaned = np.zeros_like(tissue_mask, dtype=np.uint8)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_tissue_area:
                    cleaned[labels == i] = 1

            tissue_mask = cleaned

        except Exception as e:
            print("WARNING: cv2.connectedComponentsWithStats failed, skip component filtering.")
            print("cv2 connectedComponents error:", repr(e))

    print(f"Tissue mask ratio: {np.mean(tissue_mask > 0):.6f}")

    return tissue_mask.astype(np.uint8)


def save_mask_preview(dir_output, thumb_rgb, tissue_mask):
    os.makedirs(dir_output, exist_ok=True)

    thumb_rgb = ensure_uint8_rgb(thumb_rgb)
    tissue_mask = np.asarray(tissue_mask).astype(np.uint8)

    overlay = thumb_rgb.copy()
    overlay[tissue_mask > 0] = (
        0.6 * overlay[tissue_mask > 0].astype(np.float32)
        + 0.4 * np.array([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)

    tifffile.imwrite(
        os.path.join(dir_output, "thumbnail_rgb.tif"),
        thumb_rgb,
        photometric="rgb"
    )

    tifffile.imwrite(
        os.path.join(dir_output, "thumbnail_tissue_mask.tif"),
        (tissue_mask * 255).astype(np.uint8),
        photometric="minisblack"
    )

    tifffile.imwrite(
        os.path.join(dir_output, "thumbnail_tissue_overlay.tif"),
        overlay,
        photometric="rgb"
    )


def patch_has_tissue(
    tissue_mask,
    h,
    w,
    patch_h,
    patch_w,
    scale_y,
    scale_x,
    min_tissue_ratio=0.02,
):
    h1 = int(np.floor(h * scale_y))
    h2 = int(np.ceil((h + patch_h) * scale_y))
    w1 = int(np.floor(w * scale_x))
    w2 = int(np.ceil((w + patch_w) * scale_x))

    h1 = max(0, min(h1, tissue_mask.shape[0]))
    h2 = max(0, min(h2, tissue_mask.shape[0]))
    w1 = max(0, min(w1, tissue_mask.shape[1]))
    w2 = max(0, min(w2, tissue_mask.shape[1]))

    if h2 <= h1 or w2 <= w1:
        return False

    region = tissue_mask[h1:h2, w1:w2]

    if region.size == 0:
        return False

    tissue_ratio = np.mean(region > 0)

    return tissue_ratio >= min_tissue_ratio


class WholeSlideReader:
    def __init__(self, fp_he_img):
        self.fp_he_img = str(fp_he_img)
        self.backend = None
        self.slide = None
        self.vips_img = None
        self.tif = None
        self.arr = None
        self.layout = None

        if can_open_with_openslide(self.fp_he_img):
            try:
                self.slide = openslide.OpenSlide(self.fp_he_img)
                width, height = self.slide.dimensions
                self.height = height
                self.width = width
                self.backend = "openslide"
                print("Patch reader backend: openslide")
                return
            except Exception as e:
                print("WARNING: OpenSlide patch reader failed:", repr(e))

        if pyvips is not None:
            try:
                self.vips_img = pyvips.Image.new_from_file(self.fp_he_img, access="random")
                self.height = self.vips_img.height
                self.width = self.vips_img.width
                self.backend = "pyvips"
                print("Patch reader backend: pyvips")
                return
            except Exception as e:
                print("WARNING: pyvips patch reader failed:", repr(e))

        self.tif = tifffile.TiffFile(self.fp_he_img)
        self.arr = self.tif.series[0].asarray(out="memmap")

        if self.arr.ndim == 2:
            self.height, self.width = self.arr.shape
            self.layout = "GRAY"
        elif self.arr.ndim == 3:
            if self.arr.shape[-1] in (3, 4):
                self.height, self.width = self.arr.shape[:2]
                self.layout = "HWC"
            elif self.arr.shape[0] in (3, 4):
                self.height, self.width = self.arr.shape[1], self.arr.shape[2]
                self.layout = "CHW"
            else:
                raise ValueError(f"Unsupported image shape: {self.arr.shape}")
        else:
            raise ValueError(f"Unsupported image shape: {self.arr.shape}")

        self.backend = "tifffile"
        print("Patch reader backend: tifffile")

    def read_patch(self, h, w, h2, w2):
        patch_h = h2 - h
        patch_w = w2 - w

        if self.backend == "openslide":
            patch = self.slide.read_region(
                location=(w, h),
                level=0,
                size=(patch_w, patch_h)
            )
            patch = np.asarray(patch.convert("RGB"))
            return ensure_uint8_rgb(patch)

        elif self.backend == "pyvips":
            patch = self.vips_img.crop(w, h, patch_w, patch_h)
            patch = pyvips_to_numpy_rgb(patch)
            return ensure_uint8_rgb(patch)

        elif self.backend == "tifffile":
            if self.layout == "GRAY":
                patch = self.arr[h:h2, w:w2]
            elif self.layout == "HWC":
                patch = self.arr[h:h2, w:w2, :]
            elif self.layout == "CHW":
                patch = self.arr[:, h:h2, w:w2]
                patch = np.moveaxis(patch, 0, -1)
            else:
                raise RuntimeError(f"Unknown tifffile layout: {self.layout}")

            return ensure_uint8_rgb(patch)

        else:
            raise RuntimeError("Reader is not initialized.")

    def close(self):
        if self.slide is not None:
            self.slide.close()
        if self.tif is not None:
            self.tif.close()


def crop_patches_from_tiff_with_tissue_mask(
    fp_he_img,
    dir_output,
    crop_size,
    skip_background=True,
    white_thresh=240,
    white_ratio=0.85,
    min_std=8,
    thumb_max_size=4000,
    tissue_gray_thresh=235,
    tissue_sat_thresh=20,
    tissue_kernel=5,
    min_tissue_area=64,
    min_tissue_ratio=0.02,
    save_preview=True,
):
    os.makedirs(dir_output, exist_ok=True)

    thumb_rgb, height_whole, width_whole = load_thumbnail_for_mask(
        fp_he_img,
        thumb_max_size=thumb_max_size
    )

    tissue_mask = generate_tissue_mask(
        thumb_rgb,
        gray_thresh=tissue_gray_thresh,
        sat_thresh=tissue_sat_thresh,
        morph_kernel=tissue_kernel,
        min_tissue_area=min_tissue_area,
    )

    scale_y = tissue_mask.shape[0] / height_whole
    scale_x = tissue_mask.shape[1] / width_whole

    if save_preview:
        save_mask_preview(dir_output, thumb_rgb, tissue_mask)

    print(f"Original image size: H={height_whole}, W={width_whole}")
    print(f"Thumbnail size: {thumb_rgb.shape[:2]}")
    print(f"Tissue area ratio on thumbnail: {np.mean(tissue_mask > 0):.4f}")

    reader = WholeSlideReader(fp_he_img)

    n_total_grid = 0
    n_mask_kept = 0
    n_bg_skipped = 0
    n_saved = 0
    n_read_failed = 0

    h_starts = list(range(0, height_whole, crop_size))
    w_starts = list(range(0, width_whole, crop_size))

    try:
        for h in tqdm(h_starts, desc="Cropping rows"):
            for w in w_starts:
                h2 = min(h + crop_size, height_whole)
                w2 = min(w + crop_size, width_whole)

                n_total_grid += 1

                if not patch_has_tissue(
                    tissue_mask,
                    h,
                    w,
                    h2 - h,
                    w2 - w,
                    scale_y,
                    scale_x,
                    min_tissue_ratio=min_tissue_ratio,
                ):
                    continue

                n_mask_kept += 1

                try:
                    patch = reader.read_patch(h, w, h2, w2)
                    patch = ensure_uint8_rgb(patch)
                except Exception as e:
                    n_read_failed += 1
                    print(f"WARNING: failed to read patch h={h}, w={w}: {repr(e)}")
                    continue

                if skip_background and is_mostly_background(
                    patch,
                    white_thresh=white_thresh,
                    white_ratio=white_ratio,
                    min_std=min_std,
                ):
                    n_bg_skipped += 1
                    continue

                fp_output_patch = os.path.join(dir_output, f"{h}_{w}.tif")

                try:
                    tifffile.imwrite(
                        fp_output_patch,
                        patch,
                        photometric="rgb"
                    )
                    n_saved += 1
                except Exception as e:
                    print(f"WARNING: failed to save patch {fp_output_patch}: {repr(e)}")
                    continue

    finally:
        reader.close()

    print("Done cropping with tissue mask.")
    print(f"Total grid patches: {n_total_grid}")
    print(f"Kept by tissue mask: {n_mask_kept}")
    print(f"Skipped by background filter: {n_bg_skipped}")
    print(f"Failed reading patches: {n_read_failed}")
    print(f"Final saved patches: {n_saved}")

    if n_saved == 0:
        print("WARNING: No patches saved. You may need to lower --min_tissue_ratio or adjust tissue thresholds.")

    return height_whole, width_whole


def run_command(cmd, cwd=None):
    print("\n[Running command]")
    print(" ".join(cmd))

    result = subprocess.run(cmd, cwd=cwd)

    return result.returncode


def run_hovernet(
    dir_hovernet,
    gpu_id,
    dir_crops,
    dir_out_hovernet,
    batch_file_num=50
):
    dir_crops = os.path.abspath(dir_crops)
    dir_out_hovernet = os.path.abspath(dir_out_hovernet)

    os.makedirs(dir_out_hovernet, exist_ok=True)

    fps_crops = glob.glob(os.path.join(dir_crops, "*.tif"))
    fps_crops = [
        x for x in fps_crops
        if not os.path.basename(x).startswith("thumbnail_")
    ]
    fps_crops = natsort.natsorted(fps_crops)

    print("Input crops dir:", dir_crops)
    print("HoverNet output dir:", dir_out_hovernet)
    print("Num crops found:", len(fps_crops))

    if len(fps_crops) == 0:
        print("No crop files found.")
        return

    if batch_file_num is None or batch_file_num <= 0 or batch_file_num >= len(fps_crops):
        cmd = [
            "python", "run_infer.py",
            f"--gpu={gpu_id}",
            "--nr_types=0",
            "--batch_size=32",
            "--model_mode=original",
            "--model_path=hovernet_original_consep_notype_tf2pytorch.tar",
            "--nr_inference_workers=4",
            "--nr_post_proc_workers=8",
            "tile",
            f"--input_dir={dir_crops}",
            f"--output_dir={dir_out_hovernet}",
            "--mem_usage=0.3",
        ]

        ret = run_command(cmd, cwd=dir_hovernet)

        if ret != 0:
            print(f"HoverNet failed. return code={ret}")

        return

    dir_crops_temp = dir_crops + "_temp_batches"
    os.makedirs(dir_crops_temp, exist_ok=True)

    try:
        for i in range(0, len(fps_crops), batch_file_num):
            fps_batch = fps_crops[i:i + batch_file_num]
            batch_idx = i // batch_file_num + 1

            print(f"\nProcessing batch {batch_idx}, num files = {len(fps_batch)}")

            for f in glob.glob(os.path.join(dir_crops_temp, "*.tif")):
                os.remove(f)

            for fp in fps_batch:
                dst_file = os.path.join(dir_crops_temp, os.path.basename(fp))
                shutil.copy2(fp, dst_file)

            batch_output_dir = os.path.join(
                dir_out_hovernet,
                f"batch_{batch_idx:04d}"
            )
            os.makedirs(batch_output_dir, exist_ok=True)

            cmd = [
                "python", "run_infer.py",
                f"--gpu={gpu_id}",
                "--nr_types=0",
                "--batch_size=32",
                "--model_mode=original",
                "--model_path=hovernet_original_consep_notype_tf2pytorch.tar",
                "--nr_inference_workers=4",
                "--nr_post_proc_workers=8",
                "tile",
                f"--input_dir={dir_crops_temp}",
                f"--output_dir={batch_output_dir}",
                "--mem_usage=0.3",
            ]

            ret = run_command(cmd, cwd=dir_hovernet)

            if ret != 0:
                print(f"HoverNet failed on batch {batch_idx}, return code={ret}")
                break

    finally:
        if os.path.exists(dir_crops_temp):
            shutil.rmtree(dir_crops_temp)


def combine_crops(hist_h, hist_w, dir_output, dir_hovernet_output):
    output = np.zeros((hist_h, hist_w), dtype=np.uint32)

    files = glob.glob(
        os.path.join(dir_hovernet_output, "**", "mat", "*.mat"),
        recursive=True
    )
    files = natsort.natsorted(files)

    if len(files) == 0:
        print("No .mat files found for combining.")
        return

    fp_out = os.path.join(dir_output, "he_image_nuclei_seg.tif")

    hs_all = []
    ws_all = []
    total_n = 0
    region_ids = []

    for mat_fname in files:
        coords = re.findall(r"\d+", os.path.basename(mat_fname))

        if len(coords) < 2:
            print(f"Skipping invalid mat filename: {mat_fname}")
            continue

        hs = int(coords[0])
        ws = int(coords[1])

        hs_all.append(hs)
        ws_all.append(ws)

        mat_contents = sio.loadmat(mat_fname)
        inst_map = mat_contents["inst_map"]

        he = min(hs + inst_map.shape[0], hist_h)
        we = min(ws + inst_map.shape[1], hist_w)

        inst_crop = inst_map[:he - hs, :we - ws]

        nuclei_mask = np.where(inst_crop > 0, 1, 0).astype(np.uint32)
        output[hs:he, ws:we] = inst_crop.astype(np.uint32) + total_n * nuclei_mask

        unique_ids = np.unique(inst_crop)
        unique_ids = unique_ids[unique_ids > 0]

        region_ids.extend(list(unique_ids + total_n))
        total_n += len(unique_ids)

    print(f"Num IDs before combining: {total_n}")

    if len(region_ids) == 0:
        print("No nuclei IDs found.")
        tifffile.imwrite(fp_out, output, photometric="minisblack")
        print("Saved empty segmentation:", fp_out)
        return

    region_ids = np.array(region_ids, dtype=np.uint32)

    hs_all_lower = natsort.natsorted(list(set(hs_all)))
    ws_all_right = natsort.natsorted(list(set(ws_all)))

    if 0 in hs_all_lower:
        hs_all_lower.remove(0)

    if 0 in ws_all_right:
        ws_all_right.remove(0)

    hs_all_upper = [x - 1 for x in hs_all_lower if x - 1 >= 0 and x < hist_h]
    hs_all_lower = [x for x in hs_all_lower if x < hist_h]

    ws_all_left = [x - 1 for x in ws_all_right if x - 1 >= 0 and x < hist_w]
    ws_all_right = [x for x in ws_all_right if x < hist_w]

    output2 = np.copy(output)

    print("Combining crops along horizontal borders...")

    for hsu, hsl in tqdm(
        zip(hs_all_upper, hs_all_lower),
        total=min(len(hs_all_upper), len(hs_all_lower))
    ):
        border_a = output[hsu, :].copy()
        border_b = output[hsl, :].copy()

        mask_a = border_a > 0
        mask_b = border_b > 0
        overlap = mask_a & mask_b

        overlap_locs = np.where(overlap)[0]

        if len(overlap_locs) == 0:
            continue

        overlap_ids_a = output[hsu, overlap_locs]
        overlap_ids_b = output[hsl, overlap_locs]

        d = dict(zip(overlap_ids_b, overlap_ids_a))

        h_end = min(hsl + 150, hist_h)

        for old, new in d.items():
            if old == 0 or new == 0 or old == new:
                continue
            region = output2[hsl:h_end, :]
            region_ref = output[hsl:h_end, :]
            region[region_ref == old] = new
            region_ids[region_ids == old] = new

    print(f"Num nuclei intermediate step {len(np.unique(region_ids))}")

    del output

    output3 = np.copy(output2)

    print("Combining crops along vertical borders...")

    for wsl, wsr in tqdm(
        zip(ws_all_left, ws_all_right),
        total=min(len(ws_all_left), len(ws_all_right))
    ):
        border_a = output2[:, wsl].copy()
        border_b = output2[:, wsr].copy()

        mask_a = border_a > 0
        mask_b = border_b > 0
        overlap = mask_a & mask_b

        overlap_locs = np.where(overlap)[0]

        if len(overlap_locs) == 0:
            continue

        overlap_ids_a = output2[overlap_locs, wsl]
        overlap_ids_b = output2[overlap_locs, wsr]

        d = dict(zip(overlap_ids_b, overlap_ids_a))

        w_end = min(wsr + 150, hist_w)

        for old, new in d.items():
            if old == 0 or new == 0 or old == new:
                continue
            region = output3[:, wsr:w_end]
            region_ref = output2[:, wsr:w_end]
            region[region_ref == old] = new
            region_ids[region_ids == old] = new

    print(f"Num nuclei final: {len(np.unique(region_ids))}")

    tifffile.imwrite(
        fp_out,
        output3,
        photometric="minisblack"
    )

    print("Saved", fp_out)

    resized_h = int(round(hist_h * 0.2125))
    resized_w = int(round(hist_w * 0.2125))

    if cv2 is not None:
        try:
            output_microns = cv2.resize(
                output3.astype(np.float32),
                (resized_w, resized_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint32)
        except Exception as e:
            print("WARNING: cv2.resize failed, fallback to nearest numpy resize.")
            print("cv2 resize error:", repr(e))
            y_idx = np.linspace(0, hist_h - 1, resized_h).astype(np.int64)
            x_idx = np.linspace(0, hist_w - 1, resized_w).astype(np.int64)
            output_microns = output3[np.ix_(y_idx, x_idx)].astype(np.uint32)
    else:
        y_idx = np.linspace(0, hist_h - 1, resized_h).astype(np.int64)
        x_idx = np.linspace(0, hist_w - 1, resized_w).astype(np.int64)
        output_microns = output3[np.ix_(y_idx, x_idx)].astype(np.uint32)

    fp_out_microns = fp_out.replace(".tif", "_microns.tif")

    tifffile.imwrite(
        fp_out_microns,
        output_microns,
        photometric="minisblack"
    )

    print("Saved", fp_out_microns)
    print(f"Num nuclei final in downsized copy: {len(np.unique(output_microns)) - 1}")


def main(config):
    dir_output = os.path.join(config.dir_output, "he_image_nuclei_seg_crops")
    os.makedirs(dir_output, exist_ok=True)

    dir_output_hovernet = dir_output + "_hovernet"

    if config.step == 1:
        crop_patches_from_tiff_with_tissue_mask(
            config.fp_he_img,
            dir_output,
            config.crop_size,
            skip_background=config.skip_background,
            white_thresh=config.white_thresh,
            white_ratio=config.white_ratio,
            min_std=config.min_std,
            thumb_max_size=config.thumb_max_size,
            tissue_gray_thresh=config.tissue_gray_thresh,
            tissue_sat_thresh=config.tissue_sat_thresh,
            tissue_kernel=config.tissue_kernel,
            min_tissue_area=config.min_tissue_area,
            min_tissue_ratio=config.min_tissue_ratio,
            save_preview=config.save_tissue_preview,
        )

    elif config.step == 2:
        print("Make sure you are using the correct env for hovernet")

        os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

        run_hovernet(
            config.dir_hovernet,
            config.gpu_id,
            dir_output,
            dir_output_hovernet,
            batch_file_num=config.batch_file_num,
        )

    elif config.step == 3:
        hist_h, hist_w, _ = get_image_hw_and_layout(config.fp_he_img)

        combine_crops(
            hist_h,
            hist_w,
            config.dir_output,
            dir_output_hovernet
        )

        if config.del_intm_files:
            print("Deleting intermediate files")

            if os.path.exists(dir_output):
                shutil.rmtree(dir_output)

            if os.path.exists(dir_output_hovernet):
                shutil.rmtree(dir_output_hovernet)

    else:
        sys.exit("Invalid --step specified, (1=crop H&E, 2=run Hover-Net, 3=combine crops)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fp_he_img", default="he_image.ome.tif", type=str)
    parser.add_argument("--dir_hovernet", default="hover_net", type=str)
    parser.add_argument("--crop_size", default=2000, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--dir_output", default="data_processing", type=str)
    parser.add_argument("--del_intm_files", default=True, type=str2bool)

    parser.add_argument("--skip_background", default=True, type=str2bool)
    parser.add_argument("--white_thresh", default=240, type=int)
    parser.add_argument("--white_ratio", default=0.85, type=float)
    parser.add_argument("--min_std", default=8.0, type=float)

    parser.add_argument("--thumb_max_size", default=4000, type=int)
    parser.add_argument("--tissue_gray_thresh", default=235, type=int)
    parser.add_argument("--tissue_sat_thresh", default=20, type=int)
    parser.add_argument("--tissue_kernel", default=5, type=int)
    parser.add_argument("--min_tissue_area", default=64, type=int)
    parser.add_argument("--min_tissue_ratio", default=0.02, type=float)
    parser.add_argument("--save_tissue_preview", default=True, type=str2bool)

    parser.add_argument("--batch_file_num", default=50, type=int)

    config = parser.parse_args()

    main(config)