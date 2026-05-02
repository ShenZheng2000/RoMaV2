import torch
import torch.nn.functional as F
import numpy as np
from romav2.device import device
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def vis(img1, img2, warp_AB, warp_BA, certainty_AB, certainty_BA):
    H, W = warp_AB.shape[1:3]
    x1 = F.interpolate(
        (torch.tensor(np.array(img1)) / 255).to(device).permute(2, 0, 1)[None],
        size=(H, W),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    x2 = F.interpolate(
        (torch.tensor(np.array(img2)) / 255).to(device).permute(2, 0, 1)[None],
        size=(H, W),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    im2_transfer_rgb = F.grid_sample(
        x2, warp_AB, mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
        x1, warp_BA, mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)  # .permute(1,2,0)
    white_im = torch.ones((H, 2 * W), device=device)
    certainty = torch.cat((certainty_AB, certainty_BA), dim=2)

    vis_im = certainty[0] * warp_im + (1 - certainty[0]) * white_im
    x = torch.cat((x1[0], x2[0]), dim=2)
    vis_im = torch.cat((x, vis_im), dim=1)
    return vis_im


# ── confidence / overlap visualization ────────────────────────────────────────

def save_overlap_histogram(overlap_BA, save_path):
    overlap_B = overlap_BA[..., 0].cpu().numpy().flatten()
    nonzero = overlap_B[overlap_B > 0.01]
    plt.figure(figsize=(10, 5))
    plt.hist(nonzero, bins=100, color='red', edgecolor='black')
    plt.title('Zoomed: pixels > 0.01 only')
    plt.xlabel('Overlap confidence value')
    plt.ylabel('Pixel count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def overlay_confidence_map(overlap, im_path, H, W, alpha=0.5, colormap='jet'):
    conf_np = overlap[..., 0].cpu().numpy()
    cmap = cm.get_cmap(colormap)
    conf_colored = cmap(conf_np)
    conf_colored_rgb = (conf_colored[..., :3] * 255).astype(np.uint8)
    img_orig = Image.open(im_path).convert('RGB')
    W_orig, H_orig = img_orig.size
    conf_pil = Image.fromarray(conf_colored_rgb).resize((W_orig, H_orig), Image.BILINEAR)
    return Image.blend(img_orig, conf_pil, alpha=alpha)


def visualize_confidence_maps(overlap_BA, warp_AB, im1_path, im2_path, save_path, H, W, alpha=0.5, colormap='jet'):
    blended_B = overlay_confidence_map(overlap_BA, im2_path, H, W, alpha=alpha, colormap=colormap)

    warp_AB_np = warp_AB.cpu().numpy()
    conf_B_np = overlap_BA[..., 0].cpu().numpy()
    warp_x = ((warp_AB_np[..., 0] + 1) / 2 * (W - 1)).clip(0, W - 1).astype(np.float32)
    warp_y = ((warp_AB_np[..., 1] + 1) / 2 * (H - 1)).clip(0, H - 1).astype(np.float32)
    conf_A_np = conf_B_np[warp_y.astype(int), warp_x.astype(int)]
    conf_A_tensor = torch.from_numpy(conf_A_np).unsqueeze(-1)
    blended_A = overlay_confidence_map(conf_A_tensor, im1_path, H, W, alpha=alpha, colormap=colormap)

    W_A, H_A = blended_A.size
    W_B, H_B = blended_B.size
    combined = Image.new('RGB', (W_A + W_B, max(H_A, H_B)))
    combined.paste(blended_A, (0, 0))
    combined.paste(blended_B, (W_A, 0))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(save_path))
    print(f"Saved confidence overlay to {save_path}")


def save_saliency_map(overlap_BA, grid_shape=(31, 51), pkl_path="dataset_saliency.pkl", png_path="dataset_saliency.png"):
    import pickle

    # save full-res grayscale confidence of right image (no overlay)
    sal_np = overlap_BA[..., 0].cpu().numpy()
    sal_np = (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)
    Image.fromarray((sal_np * 255).astype(np.uint8), mode='L').save(png_path)

    # save pkl (downscaled to grid, normalized to distribution)
    sal = torch.from_numpy(sal_np).unsqueeze(0).unsqueeze(0)
    sal = F.interpolate(sal, size=grid_shape, mode='bilinear', align_corners=False)
    sal = sal / sal.sum()
    pickle.dump(sal.cpu(), open(pkl_path, "wb"))

    print(f"Saved saliency map to {pkl_path} and {png_path}")