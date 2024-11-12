import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from sklearn.metrics import jaccard_score
import torch.nn as nn

def calcMetrics(true_data, eval_data, label_data, seg_data, verbose=1):
    """calculate metrics from eval and true data

    Args:
        true_data (_type_): true data in 4D tensor
        eval_data (_type_): denoised data in 4D tensor
        label_data (_type_): label data in 4D tensor
        seg_data (_type_): segmentation data in 4D tensor
        verbose (int, optional): Setting print info (0: None, 1: Overall average metrics, 2: Metrics of all images). Defaults to 1.

    Returns:
        Tuple: Tuple of four lists (PSNR, SSIM, RMSE, mIoU)
    """
    num_image = len(true_data)
    psnr, ssim, rmse, iou = [], [], [], []

    for i in range(num_image):
        # Get images
        im_true = np.squeeze(true_data[i, :, :, :]).numpy().astype(np.float32)
        im_eval = np.squeeze(eval_data[i, :, :, :]).numpy().astype(np.float32)

        im_seg = seg_data[i, :, :, :]
        sig = nn.Sigmoid()
        im_seg_sig = sig(im_seg)
        im_seg_sig = np.squeeze(im_seg_sig).numpy().astype(np.float32)
        im_seg_sig[im_seg_sig>=0.5] = 1 
        im_seg_sig[im_seg_sig<0.5] = 0
        im_label = np.squeeze(label_data[i, :, :, :]).numpy().astype(np.float32)

        
        # PSNR
        p = peak_signal_noise_ratio(im_true, im_eval)

        # RSME
        r = mean_squared_error(im_true.flatten(), im_eval.flatten(), squared=False)

        # SSIM
        val_min = im_true.min()
        val_range = im_true.max() - val_min
        im_true_norm = (im_true - val_min) / val_range
        im_eval_norm = (im_eval - val_min) / val_range
        im_max = max(im_eval_norm.max(), im_true_norm.max())
        im_min = min(im_eval_norm.min(), im_true_norm.min())
        s = structural_similarity(
            im_true_norm, im_eval_norm, data_range=im_max - im_min, channel_axis=0
        )

        # mIoU
        o = jaccard_score(im_label.flatten(), im_seg_sig.flatten(), pos_label=1, average='binary')
        
        if verbose == 2:
            print(f"#{i}: PSNR = {p:.1f}, SSIM = {s:.3f}, RMSE = {r:.3f}, IoU = {o:.3f}")

        psnr.append(p)
        ssim.append(s)
        rmse.append(r)
        iou.append(o)
    
    if verbose > 0:
        print(
            f"PSNR = {np.mean(psnr):.1f}, SSIM = {np.mean(ssim):.3f}, RMSE = {np.mean(rmse):.3f}, mIoU = {np.mean(iou):.3f}"
        )

    return psnr, ssim, rmse, iou
