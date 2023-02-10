from typing import Any, Dict, List, Mapping, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F

from src.criterion.gce_loss import generalized_cross_entropy
from .matcher import HungarianMatcher


class NerSetCriterion:
    def __init__(self,
        match_weight: Dict[str, float],
        nil_weight: float = -1.0,
        match_solver: str = "hungarian",
        boundary_loss_type = "ce",
        cls_loss_type = "ce",
        **kwargs,
    ):
        self.matcher = HungarianMatcher(cost_class=match_weight["cls"], cost_span=match_weight["boundary"], solver=match_solver)
        self.nil_weight = nil_weight
        self.boundary_loss_type = boundary_loss_type
        self.cls_loss_type = cls_loss_type
        self.kwargs = kwargs

    # @torchsnooper.snoop()
    def forward(self, outputs: Mapping[str, Any], targets: Mapping[str, Any]):
        """outputs无论是边界还是cls均需要是logits（没有激活函数），gt如果是soft label的话需要是softmax的

        Args:
            outputs (_type_): _description_
            targets (_type_): _description_

        Returns:
            _type_: _description_
        """
        pred_logits = outputs["pred_logits"]        # bsz, n_queries, n_classes
        query_nil_weight = outputs.get("query_nil_weight")
        pred_left = outputs["pred_left"]            # bsz, n_queries, sent_len
        pred_right = outputs["pred_right"]          # bsz, n_queries, sent_len
        # 因为matcher以及CE的需要，转换为flatten的形式
        gt_types: Tensor = targets["gt_types"]        # bsz, n_gt or bsz, n_gt, n_classes
        gt_masks: Tensor = targets["gt_masks"]        # bsz, n_gt
        gt_spans: Tensor = targets["gt_spans"]        # bsz, n_gt, 2 or bsz, n_gt, 2, sent_len

        cls_kl = False
        if gt_types.dim() == 3:
            # loss将会使用KLDivLoss
            cls_kl = True
            # n_gts, n_classes
            gt_types_wo_nil = torch.masked_select(
                gt_types, gt_masks.unsqueeze(2).repeat(1, 1, gt_types.size()[2])
            ).view(-1, gt_types.size()[2])
        else:
            # CE
            gt_types_wo_nil = torch.masked_select(gt_types, gt_masks)       # n_gts

        boundary_kl = False
        if gt_spans.dim() == 4:
            boundary_kl = True
            # n_gts, 2, sent_len
            spans_wo_nil = torch.masked_select(
                gt_spans, gt_masks.view(*gt_masks.size(), 1, 1).repeat(1, 1, 2, gt_spans.size()[-1])
            ).view(-1, 2, gt_spans.size()[-1])
            gt_left_wo_nil = spans_wo_nil[:, 0, :]      # n_gts, sent_len
            gt_right_wo_nil = spans_wo_nil[:, 1, :]     # n_gts, sent_len
        else:
            spans_wo_nil = torch.masked_select(gt_spans, gt_masks.unsqueeze(2).repeat(1, 1, 2)).view(-1, 2)     # n_gts, 2
            gt_left_wo_nil = spans_wo_nil[:, 0]     # n_gts
            gt_right_wo_nil = spans_wo_nil[:, 1]    # n_gts

        sizes = [i.sum() for i in gt_masks]

        indices = self.matcher.forward(outputs, {
                "gt_types": gt_types_wo_nil,
                "gt_left": gt_left_wo_nil,
                "gt_right": gt_right_wo_nil,
                "sizes": sizes
            }
        )

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_spans = sum(sizes)
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {
            "cls": self._loss_labels(
                pred_logits, gt_types_wo_nil, sizes,
                indices, num_spans,
                query_nil_weight, cls_kl,
            ),
            "boundary": self._loss_boundary(
                pred_left, pred_right, outputs["token_mask"],
                gt_left_wo_nil, gt_right_wo_nil, sizes,
                indices, num_spans, boundary_kl
            ),
        }

        return losses, indices


    def _loss_labels(
        self,
        logits: Tensor,
        targets: Tensor,
        sizes: List[int],
        indices: List[Tuple],
        num_spans,
        query_nil_weight: Optional[Tensor] = None,
        kl_loss = False,
    ):
        """Classification loss

        Args:
            logits (Tensor): (bsz, n_queries, n_classes)
            targets (Tensor): (n_gts, n_classes) if kl_loss == Ture else (n_gts)
            sizes (List[int]): _description_
            indices (List[Tuple]): _description_
            num_spans (_type_): _description_
            query_nil_weight (Optional[Tensor]): (bsz, n_queries), 会和self.nil_weight一起起作用
            kl_loss (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        idx = self._get_src_permutation_idx(indices)

        labels = torch.split(targets, sizes, dim=0)

        weight = torch.ones(logits.size()[-1], dtype=logits.dtype, device=logits.device)
        if self.nil_weight < 0 and logits.size(0) * logits.size(1) > num_spans:
            # NOTE: 这里做了一个修正，本来cond只有self.nil_weight < 0，
            # 但是在n_gts很大的时候，甚至大于等于n_candidates的时候，原来的公式会计算出inf或者负的weight，这是不合理的
            # 考虑到这个weight是在nil太多的时候（n_candidates << n_gts)平衡nil的权重，而nil比较小的时候，权重和其他（1.0）一样就好了
            if num_spans == 0:
                # 不知道为什么，num_spans是0的时候会返回nll(只bp负样本，但负样本权重是0)，这里给一个小量
                weight[0] = 1e-25
            else:
                weight[0] = num_spans / (logits.size(0) * logits.size(1) - num_spans)
        else:
            weight[0] = self.nil_weight

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(labels, indices)])
        if kl_loss:
            prob = torch.log_softmax(logits[idx], dim=-1)
            # 因为是soft label，所以我觉得不应该让没有匹配的logits也形成一个none的gt，这太hard了

            # debug_detect_value_anomaly(prob)
            # 需要target distribution是softmax的
            loss_cls = F.kl_div(prob, target_classes_o, reduction='none') * weight.unsqueeze(0)
            # NOTE: 因为soft label没有nil的概念，所以不使用nil weight
        elif self.cls_loss_type == "ce":
            # CE loss
            # matcher生成的indices并没有完全匹配（在prediction比gt多的情况下），没有匹配的那一部分的gt需要在这里手动赋为nil
            target_classes = torch.zeros(
                logits.shape[:2], dtype=torch.int64, device=logits.device, requires_grad=False
            )
            target_classes[idx] = target_classes_o

            logits = logits.view(-1, logits.size(2))
            target_classes = target_classes.view(-1)
            loss_cls = F.cross_entropy(logits, target_classes, weight, reduction='none')
        elif self.cls_loss_type == "gce":
            target_classes = torch.zeros(
                logits.shape[:2], dtype=torch.int64, device=logits.device, requires_grad=False
            )
            target_classes[idx] = target_classes_o

            logits = logits.view(-1, logits.size(2))
            target_classes = target_classes.view(-1)
            loss_cls = generalized_cross_entropy(logits, target_classes, self.kwargs["gce_q"], weight, reduction='none')
        else:
            raise NotImplementedError(self.cls_loss_type)

        if query_nil_weight is not None:
            query_weight = query_nil_weight
            query_weight[idx] = 1       # 匹配上的部分的weight是1，只有nil会使用query nil weight
            loss_cls = loss_cls * query_weight.view(-1)

        return torch.mean(loss_cls)

    def _loss_boundary(
        self,
        pred_left: Tensor, pred_right: Tensor, token_masks: Tensor,
        gt_left: Tensor, gt_right: Tensor, sizes: List[int],
        indices: List[Tuple],
        num_spans,
        kl_loss = False,
    ):
        """与cls loss不同，boundary loss只对匹配上的prediction进行BP，未匹配的prediction不产生边界loss

        Args:
            pred_left (Tensor): (bsz, n_queries, sent_len)
            pred_right (Tensor): (bsz, n_queries, sent_len)
            token_masks (Tensor): _description_
            gt_left (Tensor): (n_gts, sent_len) or (n_gts)
            gt_right (Tensor): (n_gts, sent_len) or (n_gts)
            sizes (List[int]): _description_
            indices (List[Tuple]): _description_
            num_spans (_type_): _description_
            kl_loss (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        idx = self._get_src_permutation_idx(indices)
        src_spans_left = pred_left[idx]     # (n_match, sent_len)
        src_spans_right = pred_right[idx]   # (n_match, sent_len)

        if len(src_spans_left) == 0:
            # 因为span loss只计算匹配上的，如果没有匹配直接返回0，不返回似乎会出现nan
            return torch.zeros((1), dtype=torch.float, device=pred_left.device)

        token_masks = token_masks.unsqueeze(1).expand(-1, pred_right.size(1), -1)
        token_masks = token_masks[idx]

        gt_left = gt_left.split(sizes, dim=0)       # [n_gt] or [(n_gt, sent_len)]
        target_spans_left = torch.cat([t[i] for t, (_, i) in zip(gt_left, indices)], dim=0)     # (n_gts, sent_len) or (n_gts)
        gt_right = gt_right.split(sizes, dim=0)
        target_spans_right = torch.cat([t[i] for t, (_, i) in zip(gt_right, indices)], dim=0)

        if kl_loss:
            # target_spans_left = (n_gts, sent_len)
            # left_loss = torch.zeros_like(src_spans_left)
            # right_loss = torch.zeros_like(src_spans_left)
            # 注意对于padding tokens，predictor会赋一个一个非常小的值(-1e25)使得softmax之后这些token的结果为0
            # 这导致在KLDivLoss的计算过程中会出现log(0)，进而导致梯度出现问题
            # 这里和BertTagging里面的KLDivLoss一样，做一个处理，只计算active的KLDivLoss
            flat_masks = token_masks.flatten()
            active_left = torch.log_softmax(src_spans_left, dim=-1).flatten()[flat_masks]
            active_right = torch.log_softmax(src_spans_right, dim=-1).flatten()[flat_masks]
            active_target_left = target_spans_left.flatten()[flat_masks]
            active_target_right = target_spans_right.flatten()[flat_masks]
            left_loss = F.kl_div(active_left, active_target_left, reduction='none')
            right_loss = F.kl_div(active_right, active_target_right, reduction='none')
            loss = torch.mean(left_loss + right_loss)
        else:
            # target_spans_left = (n_gts)

            if self.boundary_loss_type == "ce":
                active_left = src_spans_left
                active_right = src_spans_right
                active_target_left = target_spans_left
                active_target_right = target_spans_right
                left_loss = F.cross_entropy(src_spans_left, target_spans_left)
                right_loss = F.cross_entropy(src_spans_right, target_spans_right)
                loss = left_loss + right_loss
            elif self.boundary_loss_type == "bce":
                left_onehot = torch.zeros(
                    [target_spans_left.size(0), src_spans_left.size(1)],
                    dtype=torch.float32, device=target_spans_left.device, requires_grad=False
                )
                left_onehot.scatter_(1, target_spans_left.unsqueeze(1), 1)
                right_onehot = torch.zeros(
                    [target_spans_right.size(0), src_spans_right.size(1)],
                    dtype=torch.float32, device=target_spans_right.device, requires_grad=False
                )
                right_onehot.scatter_(1, target_spans_right.unsqueeze(1), 1)
                left_loss = F.binary_cross_entropy(torch.sigmoid(src_spans_left), left_onehot, reduction='none')
                right_loss = F.binary_cross_entropy(torch.sigmoid(src_spans_right), right_onehot, reduction='none')
                loss = (left_loss + right_loss) * token_masks
            elif self.boundary_loss_type == "bce_softmax":
                left_onehot = torch.zeros(
                    [target_spans_left.size(0), src_spans_left.size(1)],
                    dtype=torch.float32, device=target_spans_left.device, requires_grad=False
                )
                left_onehot.scatter_(1, target_spans_left.unsqueeze(1), 1)
                right_onehot = torch.zeros(
                    [target_spans_right.size(0), src_spans_right.size(1)],
                    dtype=torch.float32, device=target_spans_right.device, requires_grad=False
                )
                right_onehot.scatter_(1, target_spans_right.unsqueeze(1), 1)
                left_loss = F.binary_cross_entropy(F.softmax(src_spans_left, dim=-1), left_onehot, reduction='none')
                right_loss = F.binary_cross_entropy(F.softmax(src_spans_right, dim=-1), right_onehot, reduction='none')
                loss = (left_loss + right_loss) * token_masks
            else:
                raise NotImplementedError(self.boundary_loss_type)

            loss = torch.sum(loss) / num_spans     # 相当于mean

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
