from pathlib import Path

import torch
from PIL import Image, ImageDraw
from romav2.io import tensor_to_pil
from romav2.device import device
from romav2 import RoMaV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from romav2.vis import save_overlap_histogram, visualize_confidence_maps, save_saliency_map
import time
import pickle
import torch.nn.functional as F
from romav2.warping_layers import FixedKDEGrid, warp
from argparse import ArgumentParser


def run_and_visualize(model, im1_path, im2_path, save_path, args, H, W):
    preds = model.match(im1_path, im2_path)
    warp_AB, overlap_AB = preds["warp_AB"][0], preds["overlap_AB"][0]
    warp_BA, overlap_BA = preds["warp_BA"][0], preds["overlap_BA"][0]
    precision_AB, precision_BA = (
        preds["precision_AB"][0],
        preds["precision_BA"][0],
    )

    std_AB = torch.linalg.det(precision_AB) ** (-1 / 4)
    std_BA = torch.linalg.det(precision_BA) ** (-1 / 4)

    std_im = torch.cat((std_AB, std_BA), dim=1)
    overlap = torch.cat((overlap_AB, overlap_BA), dim=1)[..., 0]
    white_im = torch.ones((H, 2 * W), device=device)
    std_im = (std_im / args.std_max).clamp(0, 1)
    vis_im = overlap * std_im + (1 - overlap) * white_im
    if not Path(save_path).exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(vis_im).save(save_path)

    # # save histogram (SKIP for now! )
    # save_overlap_histogram(
    #     overlap_BA=overlap_BA,
    #     save_path=Path(save_path).parent / "overlap_BA_histogram.png"
    # )

    # save confidence overlay
    visualize_confidence_maps(
        overlap_BA=overlap_BA,
        warp_AB=warp_AB,
        im1_path=im1_path,
        im2_path=im2_path,
        # NOTE: this must be jpg to save space!
        save_path=Path(save_path).parent / f"confidence_overlay_a{int(args.alpha*100)}.jpg",
        H=H,
        W=W,
        alpha=args.alpha,
        colormap=args.colormap,
    )

    return overlap_BA


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/toronto_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/toronto_B.jpg", type=str)
    parser.add_argument("--save_path", default="demo/output/roma_v2_std_toronto/roma_v2_std_toronto.png", type=str)
    parser.add_argument("--std_max", default=8.0, type=float)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--colormap", default="jet", type=str)
    parser.add_argument("--attraction_fwhm", default=13, type=int)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    model = RoMaV2()
    model.apply_setting("precise")

    H, W = model.H_hr, model.W_hr

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))
    # print('img1 original size:', im1.size)
    # print('img2 original size:', im2.size)

    # ── Round 1: match, save saliency ────────────────────────────────────
    overlap_BA = run_and_visualize(model, im1_path, im2_path, save_path, args, H, W)

    pkl_path = Path(save_path).parent / "saliency.pkl"
    png_path = Path(save_path).parent / "saliency.png"
    save_saliency_map(overlap_BA, pkl_path=pkl_path, png_path=png_path)


    # ── Warp im2 using saliency ───────────────────────────────────────────
    grid_net = FixedKDEGrid(saliency_file=str(pkl_path), 
                            output_shape=(H, W), 
                            separable=True, 
                            attraction_fwhm=args.attraction_fwhm).to(device
                            )
    im2_tensor = (torch.tensor(np.array(Image.open(im2_path).resize((W, H)))) / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device)
    # print(f"H={H}, W={W}")
    # print(f"im2_tensor shape: {im2_tensor.shape}")
    grid = grid_net(im2_tensor)
    # print(f"grid shape: {grid.shape}")
    im2_warped = warp(grid, im2_tensor)
    # print(f"im2_warped shape: {im2_warped.shape}")


    # save warped im2 to round2 folder for re-matching
    # NOTE: resize back to original resolution so round2 overlay uses correct aspect ratio
    im2_warped_path = str(Path(save_path).parent / f"round2_fwhm{args.attraction_fwhm}" / "im2_warped.jpg")
    Path(im2_warped_path).parent.mkdir(parents=True, exist_ok=True)
    im2_orig_size = Image.open(im2_path).size
    tensor_to_pil(im2_warped[0]).resize(im2_orig_size).save(im2_warped_path)

    # ── Round 2: match with warped im2 ───────────────────────────────────
    save_path_round2 = str(Path(save_path).parent / f"round2_fwhm{args.attraction_fwhm}" / Path(save_path).name)
    run_and_visualize(model, im1_path, im2_warped_path, save_path_round2, args, H, W)


if __name__ == "__main__":
    main()