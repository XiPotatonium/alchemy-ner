from typing import Any, Mapping
import torch
from scipy.optimize import linear_sum_assignment
# from lapsolver import solve_dense
from .lap import auction_lap


class HungarianMatcher:
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_Any. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-Anys).
    """

    def __init__(self, cost_class: float = 1, cost_span: float = 1, solver="hungarian"):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.solver = solver

    @torch.no_grad()
    def forward(self, outputs: Mapping[str, Any], targets: Mapping[str, Any]):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           Anys in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        if self.solver == "order":
            sizes = targets["sizes"]
            indices = [(list(range(size)), list(range(size))) for size in sizes]
        else:
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            pred_logits = outputs["pred_logits"].flatten(0, 1).softmax(dim=-1)  # [bsz * n_queries, n_classes]

            # (bsz * n_queries, sent_len)
            match_left = outputs.get("match_left", outputs["pred_left"].softmax(dim=-1)).flatten(0, 1)       # 允许用特殊的match边界
            match_right = outputs.get("match_right", outputs["pred_right"].softmax(dim=-1)).flatten(0, 1)

            gt_types = targets["gt_types"]        # (n_gts) or (n_gts, n_classes), n_gt是所有batch的gt的集合
            gt_left = targets["gt_left"]        # (n_gts) or (n_gts, sent_len)
            gt_right = targets["gt_right"]

            # import pdb;pdb.set_trace()

            if tuple(gt_types.size()) == (gt_types.size()[0], pred_logits.size()[-1]):
                # 2022-07-04 你在搞毛啊，除长度有毛用？要除以模(笑死，TMD模是1)
                cost_class = -torch.einsum("qc,cg->qg", pred_logits, gt_types.transpose(0, 1)) / pred_logits.size()[-1] ** 2
            else:
                cost_class = -pred_logits[:, gt_types]      # (bsz * n_queries, n_gts)
            if tuple(gt_left.size()) == (gt_left.size()[0], match_left.size()[-1]):
                cost_left = -torch.einsum("qs,sg->qg", match_left, gt_left.transpose(0, 1)) / match_left.size()[-1] ** 2
            else:
                cost_left = -match_left[:, gt_left]
            if tuple(gt_right.size()) == (gt_right.size()[0], match_right.size()[-1]):
                cost_right = -torch.einsum("qs,sg->qg", match_right, gt_right.transpose(0, 1)) / match_right.size()[-1] ** 2
            else:
                cost_right = -match_right[:, gt_right]
            cost_span = cost_left + cost_right

            # Final cost matrix
            C = self.cost_span * cost_span + self.cost_class * cost_class

            C = C.view(bs, num_queries, -1)

            sizes = targets["sizes"]
            indices = None

            if self.solver == "hungarian":
                C = C.cpu()
                indices = []
                for i, c in enumerate(C.split(sizes, -1)):
                    indices.append(linear_sum_assignment(c[i]))
            elif self.solver == "auction":
                indices = [auction_lap(c[i])[:2] for i, c in enumerate(C.split(sizes, -1))]
            else:
                raise NotImplementedError()

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
