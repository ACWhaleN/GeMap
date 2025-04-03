import torch
import numpy as np
from torch import nn
from mmdet.models.builder import LOSSES
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from torch.nn import functional as F

@LOSSES.register_module()
class DVSLoss(nn.Module):
    """Pivotal Loss Module containing:
    - Keypoint Alignment Loss
    - Collinear Points Interpolation Loss 
    - Point Classification Loss
    """
    def __init__(self, 
                 pc_range,
                 coe_endpts=1.0,
                 collinear_pts_coe=0.5,
                 loss_weights=[3.0, 1.0, 0.2],
                 num_pts_per_vec=20):
        super().__init__()
        self.pc_range = pc_range
        self.coe_endpts = coe_endpts
        self.collinear_pts_coe = collinear_pts_coe
        self.loss_weights = loss_weights
        self.num_pts_per_vec = num_pts_per_vec

    @staticmethod
    def pivot_dynamic_matching(cost_matrix):
        """Dynamic programming based point matching (CPU numpy version)"""
        m, n = cost_matrix.shape
        min_cost = np.full((m, n), np.inf)
        mem_sort_value = np.full((m, n), np.inf)
        match_res1 = [[] for _ in range(n)]
        match_res2 = [[] for _ in range(n)]

        # Initialization
        for j in range(n - m + 1):
            match_res1[j] = [0]
            mem_sort_value[0][j] = cost_matrix[0][0]
            if j == 0:
                min_cost[0][j] = cost_matrix[0][0]

        # DP process
        for i in range(1, m):
            for j in range(i, n - m + i + 1):
                min_cost[i][j] = mem_sort_value[i-1][j-1] + cost_matrix[i][j]
                if min_cost[i][j] < mem_sort_value[i][j-1]:
                    mem_sort_value[i][j] = min_cost[i][j]
                    if i < m - 1:
                        match_res2[j] = match_res1[j-1] + [j]
                else:
                    mem_sort_value[i][j] = mem_sort_value[i][j-1]
                    if i < m - 1:
                        match_res2[j] = match_res2[j-1]
            if i < m - 1:
                match_res1, match_res2 = match_res2.copy(), [[] for _ in range(n)]

        return min_cost[-1][-1], match_res1[-2] + [n-1]

    def compute_matched_indices(self, pred_pts, gt_pts):
        """Compute matched indices via dynamic programming"""
        batch_matched_indices = []
        with torch.no_grad():
            for batch_idx in range(pred_pts.size(0)):
                img_matched = []
                cost_matrix = torch.cdist(
                    pred_pts[batch_idx].view(-1, 2),
                    gt_pts[batch_idx].view(-1, 2),
                    p=1
                ).cpu().numpy()
                _, matched_idx = self.pivot_dynamic_matching(cost_matrix)
                img_matched.append(torch.tensor(matched_idx, device=pred_pts.device))
                batch_matched_indices.append(img_matched)
        return batch_matched_indices

    def keypoint_alignment_loss(self, pred_pts, gt_pts, matched_indices):
        """Weighted L1 loss for matched keypoints"""
        alignment_loss = 0.0
        alignment_points=0
        for b in range(pred_pts.size(0)):
            matched_idx = matched_indices[b][0]  # Take first GT instance
            pred_matched = pred_pts[b][matched_idx]
            gt_matched = gt_pts[b][0][:len(matched_idx)]  # 对齐GT长度
            
            # 直接计算平均L1损失
            alignment_loss += F.l1_loss(pred_matched, gt_matched, reduction='sum')
            alignment_points += len(matched_idx)
            
        return alignment_loss / alignment_points if alignment_points > 0 else 0.0
        

    def collinear_interp_loss(self, pred_pts, gt_pts, matched_indices):
        """改进后的共线点插值损失"""
        collinear_interp_loss = 0
        collinear_interp_loss_points = 0
        
        for b in range(pred_pts.size(0)):
            # 获取排序后的匹配关键点索引
            matched_idx = torch.sort(matched_indices[b][0]).values
            if len(matched_idx) < 2:
                continue
                
            # 获取所有点索引并确定共线点
            all_indices = torch.arange(pred_pts.size(1), device=pred_pts.device)
            # 替换torch.isin为广播比较
            existing_mask = (all_indices.unsqueeze(-1) == matched_idx).any(dim=-1)
            collinear_mask = ~existing_mask
            collinear_indices = all_indices[collinear_mask]
            
            if len(collinear_indices) == 0:
                continue

            # 生成相邻关键点对
            gt_pivots = gt_pts[b][0][:len(matched_idx)]  # 对齐GT长度
            pred_pivots = pred_pts[b, matched_idx]
            
            # 收集所有插值目标
            targets = []
            predictions = []
            
            # 遍历相邻关键点对
            for i in range(len(matched_idx)-1):
                start_idx = matched_idx[i]
                end_idx = matched_idx[i+1]
                
                # 获取两个关键点之间的共线点
                between_mask = (collinear_indices > start_idx) & (collinear_indices < end_idx)
                between_points = collinear_indices[between_mask]
                
                if len(between_points) == 0:
                    continue
                
                # 计算插值系数
                R_n = len(between_points)
                thetas = torch.arange(1, R_n+1, 
                                    dtype=torch.float32,
                                    device=pred_pts.device) / (R_n + 1)
                
                # 生成插值目标
                start_gt = gt_pivots[i]
                end_gt = gt_pivots[i+1]
                interp_targets = (1 - thetas.unsqueeze(-1)) * start_gt + thetas.unsqueeze(-1) * end_gt
                
                # 收集预测和目标
                targets.append(interp_targets)
                predictions.append(pred_pts[b, between_points])
            
            if len(targets) == 0:
                continue
            # 计算损失
            targets = torch.cat(targets)
            predictions = torch.cat(predictions)
            collinear_interp_loss += F.l1_loss(predictions, targets, reduction='sum')
            collinear_interp_loss_points += len(targets)
            
        return collinear_interp_loss / collinear_interp_loss_points if collinear_interp_loss_points > 0 else 0.0
    def point_classification_loss(self, pt_logits, matched_indices):
        """改进后的关键点分类损失"""
        batch_labels = []
        for b in range(pt_logits.size(0)):
            labels = torch.zeros(pt_logits.size(1), 
                               dtype=torch.float32,
                               device=pt_logits.device)
            matched_idx = matched_indices[b][0]
            labels[matched_idx] = 1.0
            batch_labels.append(labels)
        
        # 使用二元交叉熵损失
        return F.binary_cross_entropy_with_logits(
            input=pt_logits.view(-1),
            target=torch.cat(batch_labels),
            pos_weight=torch.tensor(2.0, device=pt_logits.device)
        )


    @staticmethod
    def linear_interpolate(start, end, num_points):
        """Generate interpolated points between start and end"""
        ratios = torch.linspace(0, 1, num_points+2, device=start.device)[1:-1]
        return start + ratios.unsqueeze(-1) * (end - start)

    def forward(self, preds_dict, gt_pts):
        """前向计算"""
        pred_pts = preds_dict['pts_preds']  # [B, N, 2]
        pt_logits = preds_dict['pts_logits']  # [B, N] (单通道logits)
        
        # 计算匹配索引
        matched_indices = self.compute_matched_indices(pred_pts, gt_pts)
        
        # Calculate losses
        loss_align = self.keypoint_alignment_loss(pred_pts, gt_pts, matched_indices)
        loss_collinear = self.collinear_interp_loss(pred_pts, gt_pts, matched_indices)
        loss_pivot_cls = self.point_classification_loss(pt_logits, matched_indices)
        
        # Weighted sum
        dvs_loss = (self.loss_weights[0] * loss_align +
                     self.loss_weights[1] * loss_collinear +
                     self.loss_weights[2] * loss_pivot_cls)
        
        return {
            'loss_pts_aligned': loss_align,
            'loss_collinear': loss_collinear,
            'loss_pivot_cls': loss_pivot_cls,
            'dvs_loss': dvs_loss
        }