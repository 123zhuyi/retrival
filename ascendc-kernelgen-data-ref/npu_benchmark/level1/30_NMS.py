import torch
import torch_npu
import torch.nn as nn
import json
import os

MAX_SV = 5000  # AscendC kernel's UB-limited max survivors, aligned with both golden and NPU implementation


# First Model class: Pure PyTorch reference NMS
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor,
                max_output_size: int, iou_threshold: float,
                scores_threshold: float, pad_to_max_output_size: bool = False):
        """
        Pure PyTorch reference NMS implementation.
        Aligned with AscendC kernel: survivors are capped at MAX_SV=5000 due to UB limit.
        """
        boxes_f32 = boxes.float()
        scores_f32 = scores.float()

        score_mask = scores_f32 > scores_threshold
        filtered_boxes = boxes_f32[score_mask]
        filtered_scores = scores_f32[score_mask]
        original_indices = torch.where(score_mask)[0]

        selected_indices = torch.zeros(max_output_size, dtype=torch.int32, device=boxes.device)

        if filtered_boxes.shape[0] == 0:
            num_selected = torch.tensor(0, dtype=torch.int32, device=boxes.device)
            return selected_indices, num_selected

        sorted_indices = torch.argsort(filtered_scores, descending=True, stable=True)
        sorted_boxes = filtered_boxes[sorted_indices]
        sorted_original_indices = original_indices[sorted_indices]

        num_boxes = sorted_boxes.shape[0]
        selected_indices_list = []
        suppressed = torch.zeros(num_boxes, dtype=torch.bool, device=boxes.device)

        areas = (sorted_boxes[:, 2] - sorted_boxes[:, 0]) * (sorted_boxes[:, 3] - sorted_boxes[:, 1])

        # Align with AscendC: effective max survivors is min(max_output_size, MAX_SV)
        effective_max = min(max_output_size, MAX_SV)

        for i in range(num_boxes):
            if suppressed[i]:
                continue

            selected_indices_list.append(sorted_original_indices[i].item())

            if len(selected_indices_list) >= effective_max:
                break

            rest = torch.arange(i + 1, num_boxes, device=boxes.device)
            if rest.numel() == 0:
                break
            mask = ~suppressed[rest]
            if not mask.any():
                continue
            candidates = rest[mask]

            cur_box = sorted_boxes[i]
            cand_boxes = sorted_boxes[candidates]

            x1_inter = torch.maximum(cur_box[0].expand(cand_boxes.shape[0]), cand_boxes[:, 0])
            y1_inter = torch.maximum(cur_box[1].expand(cand_boxes.shape[0]), cand_boxes[:, 1])
            x2_inter = torch.minimum(cur_box[2].expand(cand_boxes.shape[0]), cand_boxes[:, 2])
            y2_inter = torch.minimum(cur_box[3].expand(cand_boxes.shape[0]), cand_boxes[:, 3])

            inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
            union_area = areas[i] + areas[candidates] - inter_area
            iou = inter_area / union_area.clamp(min=1e-6)

            suppress_mask = iou >= iou_threshold
            suppressed[candidates[suppress_mask]] = True

        num_selected = len(selected_indices_list)

        if num_selected > 0:
            selected_indices[:num_selected] = torch.tensor(
                selected_indices_list, dtype=torch.int32, device=boxes.device
            )

        num_selected_tensor = torch.tensor(num_selected, dtype=torch.int32, device=boxes.device)

        return selected_indices, num_selected_tensor

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        x1_inter = max(box1[0].item(), box2[0].item())
        y1_inter = max(box1[1].item(), box2[1].item())
        x2_inter = min(box1[2].item(), box2[2].item())
        y2_inter = min(box1[3].item(), box2[3].item())

        inter_width = max(0.0, x2_inter - x1_inter)
        inter_height = max(0.0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        iou = inter_area / union_area
        return iou.item()


# Second Model class: NPU-specific NMS implementation
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor,
                max_output_size: int, iou_threshold: float,
                scores_threshold: float, pad_to_max_output_size: bool = False):
        """
        NPU-specific NMS implementation via torch_npu.npu_nms_v4.
        Aligned with AscendC kernel: survivors are capped at MAX_SV=5000 due to UB limit.
        """
        iou_threshold_tensor = torch.tensor(iou_threshold, dtype=torch.float32, device=boxes.device)
        scores_threshold_tensor = torch.tensor(scores_threshold, dtype=torch.float32, device=boxes.device)

        # Align with AscendC: effective max survivors is min(max_output_size, MAX_SV)
        effective_max = min(max_output_size, MAX_SV)

        return torch_npu.npu_nms_v4(boxes, scores, effective_max, iou_threshold_tensor,
                                     scores_threshold_tensor, pad_to_max_output_size=pad_to_max_output_size)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "30_NMS.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        boxes_info = inputs[0]
        scores_info = inputs[1]
        max_output_size_info = inputs[2]
        iou_threshold_info = inputs[3]
        scores_threshold_info = inputs[4]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[boxes_info["dtype"]]

        boxes = _make_legal_boxes(boxes_info["shape"], dtype)
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            scores = torch.normal(mu, sigma, scores_info["shape"], dtype=dtype)
        else:
            scores = torch.empty(scores_info["shape"], dtype=dtype).uniform_(-5, 5)

        max_output_size = max_output_size_info["value"]
        iou_threshold = iou_threshold_info["value"]
        scores_threshold = scores_threshold_info["value"]
        input_groups.append([boxes, scores, max_output_size, iou_threshold, scores_threshold])
    return input_groups


def get_init_inputs():
    return []


def _make_legal_boxes(shape, dtype):
    """Generate random bounding boxes with x2 > x1 and y2 > y1."""
    boxes = torch.rand(shape, dtype=dtype)
    x1, x2 = boxes[..., 0].clone(), boxes[..., 2].clone()
    boxes[..., 0] = torch.min(x1, x2)
    boxes[..., 2] = torch.max(x1, x2)
    y1, y2 = boxes[..., 1].clone(), boxes[..., 3].clone()
    boxes[..., 1] = torch.min(y1, y2)
    boxes[..., 3] = torch.max(y1, y2)
    return boxes