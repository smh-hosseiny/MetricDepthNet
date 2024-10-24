import torch
from torch import nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    def __init__(self, is_metric=True, is_synthetic=False, use_laplacian=True, include_abs_rel=True):
        """
        Depth Pro Loss Function

        Args:
            is_metric (bool): If True, the dataset is metric.
            is_synthetic (bool): If True, the dataset is synthetic. If False, it's a real-world dataset.
                                 Used to decide whether to discard top 20% of error pixels.
            use_gradients (bool): If True, includes gradient losses.
            use_laplacian (bool): If True, includes Laplacian loss.
        """
        super().__init__()
        self.is_metric = is_metric
        self.is_synthetic = is_synthetic
        self.use_laplacian = use_laplacian
        self.include_abs_rel = include_abs_rel

        self.num_scales = 6

        self.lambd = 0.8



    def forward(self, C_pred, C_gt, valid_mask):
        """
        Compute the Depth Pro loss.

        Args:
            C_pred (torch.Tensor): Predicted canonical inverse depth map of shape [B, 1, H, W].
            C_gt (torch.Tensor): Ground truth canonical inverse depth map of shape [B, 1, H, W].
            valid_mask (torch.Tensor): Binary mask indicating valid pixels of shape [B, 1, H, W].

        Returns:
            torch.Tensor: Computed loss value.
        """     


        # Ensure valid_mask is detached to prevent gradients
        valid_mask = valid_mask.detach().bool()

        # Prevent negative or zero values to avoid NaNs
        eps = 1e-6
        C_pred = torch.clamp(1/C_pred, min=eps, max=100)
        C_gt = torch.clamp(1/C_gt, min=eps, max=100)


        # Apply valid mask
        C_pred = C_pred * valid_mask
        C_gt = C_gt * valid_mask

        diff_log = torch.log(C_gt[valid_mask]) - torch.log(C_pred[valid_mask]) 
        silog_loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                                self.lambd * torch.pow(diff_log.mean(), 2))
        
        # Absolute Relative Error Loss
        abs_rel_loss = 0.0
        if self.include_abs_rel:
            abs_rel_error = torch.abs(C_gt - C_pred) / (C_gt + eps)
            abs_rel_loss = abs_rel_error[valid_mask].mean()

        loss = 0.0

        # ===== 1. Depth Loss =====
        if self.is_metric:
            # Mean Absolute Error (MAE) Loss for metric datasets, discarding top 20% of errors
            abs_diff = torch.abs(C_pred - C_gt)  # [B, 1, H, W]
            B = abs_diff.shape[0]
            
            loss_MAE = 0.0
            for b in range(B):
                valid_diff = abs_diff[b][valid_mask[b]]  # Shape: [N]
                N = valid_diff.numel()
                # Determine the number of top errors to discard (20% of N)
                # k = max(int(0.2 * N), 1)                       
                # # Sort the valid differences in ascending order
                # sorted_diff, _ = torch.sort(valid_diff)
                # filtered_diff = sorted_diff[:N - k]
                filtered_diff = valid_diff
                
                if filtered_diff.numel() == 0:
                    continue  
                
                loss_MAE += filtered_diff.mean()
            
            # Average the loss over the batch
            loss_MAE /= B
            


        else:
            # For non-metric datasets, normalize via mean absolute deviation from the median
            B = C_gt.shape[0]
            C_pred_norm = torch.zeros_like(C_pred)
            C_gt_norm = torch.zeros_like(C_gt)
            for b in range(B):
                C_pred_b = C_pred[b][valid_mask[b]]
                C_gt_b = C_gt[b][valid_mask[b]]
                if C_pred_b.numel() == 0:
                    continue
                median_pred = C_pred_b.median()
                median_gt = C_gt_b.median()
                mad_pred = torch.mean(torch.abs(C_pred_b - median_pred))
                mad_gt = torch.mean(torch.abs(C_gt_b - median_gt))
                C_pred_norm_b = (C_pred[b] - median_pred) / (mad_pred + eps)
                C_gt_norm_b = (C_gt[b] - median_gt) / (mad_gt + eps)
                C_pred_norm[b] = C_pred_norm_b
                C_gt_norm[b] = C_gt_norm_b
            # Compute MAE loss between normalized predictions and ground truths
            abs_diff = torch.abs(C_pred_norm - C_gt_norm)
            valid_pixels = valid_mask.sum()
            loss_MAE = abs_diff.sum() / (valid_pixels + eps)
            loss += loss_MAE



        # ===== 2. Derivative Losses =====
        loss_derivatives = 0.0
        if self.use_laplacian:
            # Compute multi-scale derivative losses
            C_pred_pyramid = [C_pred]
            C_gt_pyramid = [C_gt]
            valid_mask_pyramid = [valid_mask]
            for s in range(1, self.num_scales):
                # Downsample by factor of 2 per scale
                C_pred_s = F.avg_pool2d(C_pred_pyramid[-1], kernel_size=2, stride=2)
                C_gt_s = F.avg_pool2d(C_gt_pyramid[-1], kernel_size=2, stride=2)
                valid_mask_s = F.avg_pool2d(valid_mask_pyramid[-1].float(), kernel_size=2, stride=2) > 0.5
                C_pred_pyramid.append(C_pred_s)
                C_gt_pyramid.append(C_gt_s)
                valid_mask_pyramid.append(valid_mask_s)

            
            for s in range(self.num_scales):
                C_pred_s = C_pred_pyramid[s]
                C_gt_s = C_gt_pyramid[s]
                valid_mask_s = valid_mask_pyramid[s]
                # Apply valid mask
                C_pred_s = C_pred_s * valid_mask_s
                C_gt_s = C_gt_s * valid_mask_s
              
                # Laplacian kernel
                laplacian_kernel = torch.tensor([[0, 1, 0],
                                                    [1, -4, 1],
                                                    [0, 1, 0]], dtype=C_pred.dtype, device=C_pred.device).view(1, 1, 3, 3)
                # Compute Laplacians
                lap_pred = F.conv2d(C_pred_s, laplacian_kernel, padding=1)
                lap_gt = F.conv2d(C_gt_s, laplacian_kernel, padding=1)
                # Apply valid mask (excluding borders)
                valid_mask_lap = valid_mask_s[:, :, 1:-1, 1:-1]
                lap_pred = lap_pred[:, :, 1:-1, 1:-1] * valid_mask_lap
                lap_gt = lap_gt[:, :, 1:-1, 1:-1] * valid_mask_lap
                # Mean Absolute Laplacian Error (MALE)
                abs_diff_lap = torch.abs(lap_pred - lap_gt)
                valid_pixels = valid_mask_lap.sum()
                loss_MALE_s = abs_diff_lap.sum() / (valid_pixels + eps)
                loss_derivatives += loss_MALE_s
            loss_derivatives /= self.num_scales


        # Define weights
        alpha = 1  # Weight for MAE
        beta = 0  # Weight for SiLog Loss
        gamma = 0.01 # Weight for Derivative Loss

        # print(f"SiLog Loss: {silog_loss.item()}, MAE Loss: {loss_MAE.item()}, Derivative Loss: {loss_derivatives.item()}, AbsRel Loss: {abs_rel_loss.item()}")
        loss = alpha * loss_MAE + beta * silog_loss + gamma * loss_derivatives 

        return loss
