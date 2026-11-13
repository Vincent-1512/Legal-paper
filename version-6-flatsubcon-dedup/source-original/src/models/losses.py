import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss phẳng chuẩn Google 100% (Khosla et al., NeurIPS 2020).
    Hoàn toàn KHÔNG có hệ số alpha.
    
    Quy tắc phân lớp nhị phân:
      - Cặp Dương (Positive pairs): Cùng chính xác nhãn BIO (y_i == y_j) -> Trọng số 1.0.
      - Cặp Âm (Negative pairs): Bất kỳ token nào khác nhãn (y_i != y_j) -> Đẩy dạt ở mẫu số.
      - Loại trừ chính nó (self-contrast) bằng self_mask.
    """
    def __init__(self, temperature: float = 0.07):
        super(TokenSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N_entity, proj_dim] (ví dụ: [N, 128])
            labels:   [N_entity] (1D tensor chứa nhãn BIO của token)
        """
        if features.size(0) < 2:
            return features.new_tensor(0.0)

        # 1. Ép kiểu sang float32 để ổn định số học trong môi trường AMP FP16
        features = features.float()
        features = F.normalize(features, p=2, dim=1)

        # 2. Ma trận tương đồng Cosine chuẩn hóa theo nhiệt độ tau
        similarity = torch.matmul(features, features.T) / self.temperature
        N = features.size(0)

        # 3. Mặt nạ loại trừ chính nó (self-mask)
        self_mask = torch.eye(N, device=features.device, dtype=torch.bool)

        # 4. Mặt nạ mẫu dương chuẩn Google: 1.0 nếu cùng nhãn, 0.0 nếu khác nhãn (KHÔNG CÓ ALPHA)
        labels_col = labels.contiguous().view(-1, 1)
        pos_mask = (torch.eq(labels_col, labels_col.T) & (~self_mask)).float()

        # 5. Khử đường chéo mẫu số bằng -1e9 (tránh 0.0 * (-inf) = NaN)
        similarity = similarity.masked_fill(self_mask, -1e9)

        # 6. Ổn định số thực: trừ max trên mỗi hàng
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # 7. Tính xác suất log trên toàn bộ mẫu số
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        # 8. Gán đường chéo bằng 0.0
        log_prob = log_prob.masked_fill(self_mask, 0.0)

        # 9. Lọc các anchor hợp lệ có ít nhất 1 mẫu dương trong batch
        pos_count = pos_mask.sum(dim=1)
        valid_anchor = pos_count > 0

        if not valid_anchor.any():
            return features.new_tensor(0.0)

        # 10. Kỳ vọng log-likelihood trung bình trên các cặp dương (chuẩn SupContrast)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1e-12)

        # 11. Hàm mất mát SupCon scalar
        loss = -mean_log_prob_pos[valid_anchor].mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return features.new_tensor(0.0)
        return loss
