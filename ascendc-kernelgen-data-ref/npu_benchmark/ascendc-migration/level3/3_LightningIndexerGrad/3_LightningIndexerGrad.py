import torch
import torch.nn as nn
import json
import os
import math


def _safe_gather(key, topk_index, S1):
    """
    Gather key values along S2 dimension using topk_index.

    Args:
        key: [B, S2, N2, D] - key tensor
        topk_index: [B, S1, N2, K] - indices to gather
        S1: query sequence length

    Returns:
        key_topk: [B, S1, N2, K, D] - gathered key values
    """
    S2 = key.size(1)
    mask = (topk_index < 0) | (topk_index >= S2)
    safe_index = topk_index.clamp(min=0, max=S2 - 1)

    # key: [B, S2, N2, D] -> expand to [B, S1, S2, N2, D]
    key_expanded = key.unsqueeze(1).expand(-1, S1, -1, -1, -1)

    # safe_index: [B, S1, N2, K] -> expand to [B, S1, N2, K, D]
    safe_index = safe_index.unsqueeze(-1).expand(-1, -1, -1, -1, key.size(-1))
    # Permute to [B, S1, K, N2, D] for gather along dim=2
    safe_index = safe_index.permute(0, 1, 3, 2, 4)

    # gather along dim=2 (S2 dimension)
    key_topk = torch.gather(key_expanded, 2, safe_index)
    # Permute back to [B, S1, N2, K, D]
    key_topk = key_topk.permute(0, 1, 3, 2, 4)

    # mask: [B, S1, N2, K] -> expand to [B, S1, N2, K, D]
    mask = mask.unsqueeze(-1).expand(-1, -1, -1, -1, key.size(-1))
    key_topk = key_topk.masked_fill(mask, 0.0)

    return key_topk


def _tsoftmax(x):
    # Handle case where all values are 0 or -inf
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    # If x_max is -inf (all values are -inf), set to 0 to avoid NaN
    x_max = torch.where(torch.isinf(x_max), torch.zeros_like(x_max), x_max)
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    # Avoid division by zero
    x_sum = torch.where(x_sum == 0, torch.ones_like(x_sum), x_sum)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def _deal_invalid_token(B, S1, S2, topKSize, src_tensor):
    dst_tensor = src_tensor.clone()
    for b_idx in range(B):
        for s1_idx in range(S1):
            s2_realsize = int((S2 - S1) + s1_idx + 1)
            if s2_realsize <= 0:
                s2_realsize = S2
            invalid_len = topKSize - s2_realsize
            if invalid_len > 0:
                dst_tensor[b_idx, s1_idx, :, s2_realsize:] = -float('inf')
    return dst_tensor


def _safe_scatter_add_n2(beforeScatterAdd, sparse_indices, d_key_index):
    """
    Scatter add with N2 dimension support.

    Args:
        beforeScatterAdd: [B, S1, N2, K, D_index]
        sparse_indices: [B, S1, N2, K]
        d_key_index: [B, S1, S2, N2, D_index]
    """
    B, S1, S2, N2, D_index = d_key_index.shape
    mask = (sparse_indices >= 0) & (sparse_indices < S2)
    safe_indices = sparse_indices.clone()
    safe_indices[~mask] = 0
    # index_expanded: [B, S1, N2, K, D_index]
    index_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D_index)
    # Permute to [B, S1, K, N2, D_index] for scatter_add along dim=2
    index_expanded = index_expanded.permute(0, 1, 3, 2, 4)
    # masked_beforeScatterAdd: [B, S1, N2, K, D_index]
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, -1, D_index)
    masked_beforeScatterAdd = beforeScatterAdd * mask_expanded
    # Permute to [B, S1, K, N2, D_index] for scatter_add along dim=2
    masked_beforeScatterAdd = masked_beforeScatterAdd.permute(0, 1, 3, 2, 4)
    # scatter_add along dim=2 (S2 dimension)
    d_key_index.scatter_add_(2, index_expanded, masked_beforeScatterAdd)
    return d_key_index


def _safe_scatter_add(beforeScatterAdd, sparse_indices, d_key_index):
    """Legacy version for 4D tensors."""
    B, S1, S2, D_index = d_key_index.shape
    mask = (sparse_indices >= 0) & (sparse_indices < S2)
    safe_indices = sparse_indices.clone()
    safe_indices[~mask] = 0
    index_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, -1, D_index)
    masked_beforeScatterAdd = beforeScatterAdd * mask.unsqueeze(-1)
    d_key_index.scatter_add_(2, index_expanded, masked_beforeScatterAdd)
    return d_key_index


class Model(nn.Module):
    r"""
    aclnnSparseLightningIndexerGradKLLoss

    Backward pass of sparse lightning indexer with fused KL loss.
    No torch or torch_npu Python API exists; only CANN aclnn interface.
    Golden reference implemented entirely in PyTorch.

    Equivalent PyTorch native implementation (BSND):
    .. code-block:: python
        def sparse_lightning_indexer_grad_kl_loss_native(
            query, key, queryIndex, keyIndex, weights, sparseIndices,
            queryRope, keyRope, scaleValue, layout
        ):
            # 1. Merge RoPE: q_all = cat(query, queryRope), k_all = cat(key, keyRope)
            # 2. Safe gather top-k keys by sparseIndices
            # 3. Attention scores: matmul(q_all, k_topk) * scaleValue
            # 4. tsoftmax + reduce G dimension
            # 5. Index scores: matmul(queryIndex, k_index_topk) -> relu -> weighted -> softmax
            # 6. KL Loss between attention prob and index prob
            # 7. Backprop: dW, d_query_index, d_key_index via scatter_add
            return d_query_index, d_key_index, d_weights, loss, p_max, p_sum

    Args:
        query (Tensor): Query tensor [B, S1, N1, D] or [T1, N1, D].
        key (Tensor): Key tensor [B, S2, N2, D] or [T2, N2, D].
        queryIndex (Tensor): Query index tensor [B, S1, Nidx1, Didx] or [T1, Nidx1, Didx].
        keyIndex (Tensor): Key index tensor [B, S2, Nidx2, Didx] or [T2, Nidx2, Didx].
        weights (Tensor): Weights tensor [B, S1, Nidx1] or [T1, Nidx1].
        sparseIndices (Tensor): Sparse top-k indices [B, S1, Nidx2, K] or [T1, Nidx2, K].
        softmaxMax (Tensor): Softmax max from forward [B, N2, S1, G] or [N2, T1, G].
        softmaxSum (Tensor): Softmax sum from forward [B, N2, S1, G] or [N2, T1, G].
        queryRope (Tensor): Query RoPE tensor [B, S1, N1, Drope] or [T1, N1, Drope].
        keyRope (Tensor): Key RoPE tensor [B, S2, N2, Drope] or [T2, N2, Drope].
        actualSeqLengthsQ (list): Actual sequence lengths of query per batch.
        actualSeqLengthsK (list): Actual sequence lengths of key per batch.
        scaleValue (float): Attention scale factor, typically 1/sqrt(D).
        layout (str): "BSND" or "TND".

    Returns:
        tuple: (d_query_index, d_key_index, d_weights, loss, p_max, p_sum)

    Example::

        >>> query = torch.randn(1, 4, 32, 512, dtype=torch.float16)
        >>> key = torch.randn(1, 8, 1, 512, dtype=torch.float16)
        >>> qIdx = torch.randn(1, 4, 8, 128, dtype=torch.float16)
        >>> kIdx = torch.randn(1, 8, 1, 128, dtype=torch.float16)
        >>> w = torch.ones(1, 4, 8, dtype=torch.float16)
        >>> sparse = torch.randint(0, 8, (1, 4, 1, 1024), dtype=torch.int32)
        >>> smMax = torch.randn(1, 1, 4, 32, dtype=torch.float32)
        >>> smSum = torch.randn(1, 1, 4, 32, dtype=torch.float32)
        >>> qRope = torch.randn(1, 4, 32, 64, dtype=torch.float16)
        >>> kRope = torch.randn(1, 8, 1, 64, dtype=torch.float16)
        >>> Model()(query, key, qIdx, kIdx, w, sparse, smMax, smSum,
        ...          qRope, kRope, [4], [8], 0.0442, "BSND")
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, queryIndex: torch.Tensor,
                keyIndex: torch.Tensor, weights: torch.Tensor, sparseIndices: torch.Tensor,
                softmaxMax: torch.Tensor, softmaxSum: torch.Tensor, queryRope: torch.Tensor,
                keyRope: torch.Tensor, actualSeqLengthsQ: list, actualSeqLengthsK: list,
                scaleValue: float, layout: str) -> tuple:
        """
        Sparse lightning indexer backward with KL loss.

        Args:
            query (torch.Tensor): Query.
            key (torch.Tensor): Key.
            queryIndex (torch.Tensor): Query index.
            keyIndex (torch.Tensor): Key index.
            weights (torch.Tensor): Weights.
            sparseIndices (torch.Tensor): Sparse indices.
            softmaxMax (torch.Tensor): Softmax max (used by NPU, golden recompute).
            softmaxSum (torch.Tensor): Softmax sum (used by NPU, golden recompute).
            queryRope (torch.Tensor): Query RoPE.
            keyRope (torch.Tensor): Key RoPE.
            actualSeqLengthsQ (list): Query seq lengths.
            actualSeqLengthsK (list): Key seq lengths.
            scaleValue (float): Scale.
            layout (str): Layout.

        Returns:
            tuple: (d_query_index, d_key_index, d_weights, loss, p_max, p_sum).
        """
        # Ensure all tensors are on the same device as query
        device = query.device
        if key.device != device:
            key = key.to(device)
        if queryIndex.device != device:
            queryIndex = queryIndex.to(device)
        if keyIndex.device != device:
            keyIndex = keyIndex.to(device)
        if weights.device != device:
            weights = weights.to(device)
        if sparseIndices.device != device:
            sparseIndices = sparseIndices.to(device)
        if softmaxMax.device != device:
            softmaxMax = softmaxMax.to(device)
        if softmaxSum.device != device:
            softmaxSum = softmaxSum.to(device)
        if queryRope.device != device:
            queryRope = queryRope.to(device)
        if keyRope.device != device:
            keyRope = keyRope.to(device)

        # Sanitize inputs: replace NaN/Inf with 0 to avoid NaN propagation
        # This handles edge cases where inputs intentionally contain special values
        query = torch.where(torch.isfinite(query), query, torch.zeros_like(query))
        key = torch.where(torch.isfinite(key), key, torch.zeros_like(key))
        queryIndex = torch.where(torch.isfinite(queryIndex), queryIndex, torch.zeros_like(queryIndex))
        keyIndex = torch.where(torch.isfinite(keyIndex), keyIndex, torch.zeros_like(keyIndex))
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        queryRope = torch.where(torch.isfinite(queryRope), queryRope, torch.zeros_like(queryRope))
        keyRope = torch.where(torch.isfinite(keyRope), keyRope, torch.zeros_like(keyRope))

        if layout == "TND":
            return self._forward_tnd(query, key, queryIndex, keyIndex, weights, sparseIndices,
                                     softmaxMax, softmaxSum, queryRope, keyRope,
                                     actualSeqLengthsQ, actualSeqLengthsK, scaleValue)
        else:
            return self._forward_bsnd(query, key, queryIndex, keyIndex, weights, sparseIndices,
                                      softmaxMax, softmaxSum, queryRope, keyRope,
                                      actualSeqLengthsQ, actualSeqLengthsK, scaleValue)

    def _forward_bsnd(self, query, key, queryIndex, keyIndex, weights, sparseIndices,
                      softmaxMax, softmaxSum, queryRope, keyRope,
                      actualSeqLengthsQ, actualSeqLengthsK, scaleValue):
        input_dtype = query.dtype
        q_all = torch.cat((query, queryRope), dim=-1).to(input_dtype)
        k_all = torch.cat((key, keyRope), dim=-1).to(input_dtype)

        B, S1, S2, N1, N2, G, Nqi, Nki, G_index, D, D_index, topKSize = self._get_case_info(
            q_all, k_all, queryIndex, keyIndex, sparseIndices
        )

        # ========== Attention Path ==========
        # Use float32 for matmul to avoid overflow, then cast back
        query_t = q_all.to(torch.float32)
        key_t = k_all.to(torch.float32)

        # sparseIndices: [B, S1, N2, K]
        sparse_idx = sparseIndices.to(torch.int64)

        # key_topk: [B, S1, N2, K, D]
        key_topk = _safe_gather(key_t, sparse_idx, S1)

        # query_t: [B, S1, N1, D] where N1 = N2 * G
        # Reshape to [B, S1, N2, G, D] for matmul with key_topk
        query_reshaped = query_t.reshape(B, S1, N2, G, D)
        # key_topk: [B, S1, N2, K, D] -> permute to [B, S1, N2, D, K]
        key_permuted = key_topk.permute(0, 1, 2, 4, 3)
        # matmul: [B, S1, N2, G, D] @ [B, S1, N2, D, K] -> [B, S1, N2, G, K]
        p = torch.matmul(query_reshaped, key_permuted)
        p = p * scaleValue
        p = _deal_invalid_token(B, S1, S2, topKSize, p)
        pSoftmaxRes, p_max, p_sum = _tsoftmax(p)

        # pSoftmaxRes: [B, S1, N2, G, K] -> sum over G -> [B, S1, N2, K]
        pReduceGRes = torch.sum(pSoftmaxRes, dim=3)
        pReduceGRes = pReduceGRes / G
        # Cast back to input dtype
        pReduceGRes = pReduceGRes.to(input_dtype)

        # ========== Index Path ==========
        # Use float32 for matmul to avoid overflow
        query_idx_t = queryIndex.to(torch.float32)
        key_idx_t = keyIndex.to(torch.float32)

        # key_index_topk: [B, S1, Nki, K, D_index]
        key_index_topk = _safe_gather(key_idx_t, sparse_idx, S1)

        # query_idx_t: [B, S1, Nqi, Didx] where Nqi = Nki * G_index
        # Reshape to [B, S1, Nki, G_index, Didx]
        query_idx_reshaped = query_idx_t.reshape(B, S1, Nki, G_index, D_index)
        # key_index_topk: [B, S1, Nki, K, Didx] -> permute to [B, S1, Nki, Didx, K]
        key_index_permuted = key_index_topk.permute(0, 1, 2, 4, 3)
        # matmul: [B, S1, Nki, G_index, Didx] @ [B, S1, Nki, Didx, K] -> [B, S1, Nki, G_index, K]
        s = torch.matmul(query_idx_reshaped, key_index_permuted)
        # s: [B, S1, Nki, G_index, K]
        relu_res = torch.relu(s)

        # weights: [B, S1, Nqi] -> reshape to [B, S1, Nki, G_index]
        weight_reshaped = weights.reshape(B, S1, Nki, G_index)
        # Apply weights: [B, S1, Nki, G_index, K] * [B, S1, Nki, G_index, 1]
        weight_expanded = weight_reshaped.unsqueeze(-1).to(torch.float32)
        s_relu_res = relu_res * weight_expanded

        # Sum over G_index -> [B, S1, Nki, K]
        sReduceGRes = torch.sum(s_relu_res, dim=3)
        sReduceGRes = sReduceGRes / G_index
        sReduceGRes = _deal_invalid_token(B, S1, S2, topKSize, sReduceGRes)
        # Keep sReduceGRes in float32 for _tsoftmax to avoid float16 overflow
        # (values can exceed 65504 before softmax normalization)
        sSoftmaxRes, s_max, s_sum = _tsoftmax(sReduceGRes)
        # Cast softmax result to input dtype
        sSoftmaxRes = sSoftmaxRes.to(input_dtype)

        # ========== KL Loss ==========
        # Use float32 for loss computation to avoid float16 underflow
        pReduceGRes_f32 = pReduceGRes.to(torch.float32)
        sSoftmaxRes_f32 = sSoftmaxRes.to(torch.float32)
        min_value = torch.tensor([1e-8], device=pReduceGRes.device, dtype=torch.float32)
        p_clip = torch.max(pReduceGRes_f32, min_value)
        s_clip = torch.max(sSoftmaxRes_f32, min_value)
        log_p = torch.log(p_clip)
        log_s = torch.log(s_clip)
        sub_res = log_p - log_s
        mul_res = sub_res * pReduceGRes_f32
        loss = torch.sum(mul_res)
        # Keep loss in float32 to avoid float16 overflow (loss can exceed 65504)

        # ========== Backward ==========
        q_sub_p = sSoftmaxRes - pReduceGRes

        # Cast to float32 for matmul to avoid overflow
        q_sub_p_f32 = q_sub_p.to(torch.float32)
        relu_res_f32 = relu_res.to(torch.float32)
        weight_expanded_f32 = weight_expanded.to(torch.float32)
        key_index_topk_f32 = key_index_topk.to(torch.float32)
        query_idx_reshaped_f32 = query_idx_reshaped.to(torch.float32)

        # d_weight: [B, S1, Nki, K] @ [B, S1, Nki, G_index, K].permute
        # q_sub_p: [B, S1, Nki, K]
        # relu_res: [B, S1, Nki, G_index, K]
        # For each Nki, we need to compute d_weight for each G_index
        d_weight_per_group = torch.matmul(q_sub_p_f32.unsqueeze(3), relu_res_f32.permute(0, 1, 2, 4, 3))
        # d_weight_per_group: [B, S1, Nki, 1, G_index] -> squeeze -> [B, S1, Nki, G_index]
        d_weight = d_weight_per_group.squeeze(3)
        # Reshape back to [B, S1, Nqi]
        d_weight = d_weight.reshape(B, S1, Nqi)
        # Clamp to float16 range before casting to avoid overflow to inf
        if input_dtype == torch.float16:
            d_weight = d_weight.clamp(-65504, 65504)
        d_weight = d_weight.to(input_dtype)

        # relu_grad: [B, S1, Nki, G_index, K]
        relu_grad = q_sub_p_f32.unsqueeze(3) * weight_expanded_f32 * (s > 0).float()

        # d_query_index: [B, S1, Nki, G_index, K] @ [B, S1, Nki, K, Didx]
        # key_index_topk is already [B, S1, Nki, K, Didx], no permute needed
        d_query_index = torch.matmul(relu_grad, key_index_topk_f32)
        d_query_index = d_query_index.reshape(B, S1, Nqi, D_index)
        d_query_index = d_query_index.to(input_dtype)

        # before_scatter: [B, S1, Nki, K, Didx]
        # query_idx_reshaped.permute: [B, S1, Nki, Didx, G_index]
        # relu_grad: [B, S1, Nki, G_index, K]
        # matmul: [B, S1, Nki, Didx, G_index] @ [B, S1, Nki, G_index, K] -> [B, S1, Nki, Didx, K]
        before_scatter = torch.matmul(query_idx_reshaped_f32.permute(0, 1, 2, 4, 3), relu_grad)
        # Permute to [B, S1, Nki, K, Didx]
        before_scatter = before_scatter.permute(0, 1, 2, 4, 3)
        # Sum over Nki dimension to get [B, S1, K, Didx]
        before_scatter = before_scatter.sum(2)
        # Expand to [B, S1, N2, K, Didx] for N2-aware scatter_add
        before_scatter_expanded = before_scatter.unsqueeze(2).expand(-1, -1, N2, -1, -1)
        before_scatter_expanded = before_scatter_expanded.to(input_dtype)

        # d_key_index: [B, S1, S2, N2, D_index]
        d_key_index_before = torch.zeros(B, S1, S2, N2, D_index, dtype=input_dtype, device=before_scatter.device)
        d_key_index = _safe_scatter_add_n2(before_scatter_expanded, sparse_idx, d_key_index_before)
        # Sum over S1 -> [B, S2, N2, D_index]
        # Use float32 for sum to avoid float16 overflow
        d_key_index = d_key_index.to(torch.float32).sum(1)
        # Clamp to float16 range before casting back
        if input_dtype == torch.float16:
            d_key_index = d_key_index.clamp(-65504, 65504)
        d_key_index = d_key_index.to(input_dtype)
        # Reshape to match keyIndex shape: [B, S2, Nki, D_index] where Nki = N2
        # d_key_index is already [B, S2, N2, D_index]

        p_max = p_max.squeeze(0).squeeze(-1)
        p_sum = p_sum.squeeze(0).squeeze(-1)

        return d_query_index, d_key_index, d_weight, loss, p_max, p_sum

    def _forward_tnd(self, query, key, queryIndex, keyIndex, weights, sparseIndices,
                     softmaxMax, softmaxSum, queryRope, keyRope,
                     actualSeqLengthsQ, actualSeqLengthsK, scaleValue):
        B = len(actualSeqLengthsQ)
        T1 = query.shape[0]
        T2 = key.shape[0]

        d_query_index = torch.zeros(T1, queryIndex.shape[1], queryIndex.shape[2],
                                      dtype=query.dtype, device=query.device)
        d_key_index = torch.zeros(T2, keyIndex.shape[1], keyIndex.shape[2],
                                    dtype=key.dtype, device=key.device)
        d_weight = torch.zeros(T1, queryIndex.shape[1], dtype=weights.dtype, device=weights.device)
        loss = torch.zeros(1, dtype=torch.float32, device=query.device)
        p_max_all = None
        p_sum_all = None

        cum_q = [0] + list(torch.cumsum(torch.tensor(actualSeqLengthsQ), 0).numpy())
        cum_k = [0] + list(torch.cumsum(torch.tensor(actualSeqLengthsK), 0).numpy())

        for b in range(B):
            q_s, q_e = cum_q[b], cum_q[b + 1]
            k_s, k_e = cum_k[b], cum_k[b + 1]
            if q_e <= q_s:
                continue

            q_i = query[q_s:q_e].unsqueeze(0)
            k_i = key[k_s:k_e].unsqueeze(0)
            qIdx_i = queryIndex[q_s:q_e].unsqueeze(0)
            kIdx_i = keyIndex[k_s:k_e].unsqueeze(0)
            w_i = weights[q_s:q_e].unsqueeze(0)
            qRope_i = queryRope[q_s:q_e].unsqueeze(0)
            kRope_i = keyRope[k_s:k_e].unsqueeze(0)
            sparse_i = sparseIndices[q_s:q_e].unsqueeze(0)

            d_q, d_k, d_w, loss_i, p_max, p_sum = self._forward_bsnd(
                q_i, k_i, qIdx_i, kIdx_i, w_i, sparse_i,
                softmaxMax, softmaxSum, qRope_i, kRope_i,
                [actualSeqLengthsQ[b]], [actualSeqLengthsK[b]], scaleValue
            )

            d_query_index[q_s:q_e] = d_q.squeeze(0)
            d_key_index[k_s:k_e] = d_k.squeeze(0)
            d_weight[q_s:q_e] = d_w.squeeze(0)
            loss[0] += loss_i

            if p_max_all is None:
                p_max_all = p_max
                p_sum_all = p_sum
            else:
                p_max_all = torch.cat((p_max_all, p_max), dim=0)
                p_sum_all = torch.cat((p_sum_all, p_sum), dim=0)

        return d_query_index, d_key_index, d_weight, loss, p_max_all, p_sum_all

    @staticmethod
    def _get_case_info(q, k, q_index, k_index, sparse_indices):
        B = q.shape[0]
        S1 = q.shape[1]
        S2 = k.shape[1]
        N1 = q.shape[2]
        N2 = k.shape[2]
        G = N1 // N2
        Nqi = q_index.shape[2]
        Nki = k_index.shape[2]
        G_index = Nqi // Nki
        D = q.shape[3]
        D_index = q_index.shape[3]
        topKSize = sparse_indices.shape[3]
        return B, S1, S2, N1, N2, G, Nqi, Nki, G_index, D, D_index, topKSize


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "3_LightningIndexerGrad.json")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
    }

    fill_map = {
        'inf': float('inf'),
        '-inf': float('-inf'),
        'nan': float('nan'),
        '-nan': -float('nan'),
        'zero': 0.0,
    }

    input_groups = []
    # Fix random seed for deterministic validation results
    torch.manual_seed(42)

    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        scene = case.get("scene", "")

        def get_tensor(info, idx):
            dtype = dtype_map[info["dtype"]]
            shape = info["shape"]
            fill = info.get("fill", None)
            if fill in fill_map:
                return torch.full(shape, fill_map[fill], dtype=dtype)
            elif idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                raw = torch.normal(mu, sigma, shape, dtype=torch.float32)
                return raw.to(dtype)
            else:
                raw = torch.empty(shape, dtype=torch.float32).uniform_(-5, 5)
                return raw.to(dtype)

        # 14 inputs according to generator
        query = get_tensor(inputs[0], idx)
        key = get_tensor(inputs[1], idx)
        queryIndex = get_tensor(inputs[2], idx)
        keyIndex = get_tensor(inputs[3], idx)
        weights = get_tensor(inputs[4], idx)
        sparseIndices = get_tensor(inputs[5], idx)
        softmaxMax = get_tensor(inputs[6], idx)
        softmaxSum = get_tensor(inputs[7], idx)
        queryRope = get_tensor(inputs[8], idx)
        keyRope = get_tensor(inputs[9], idx)

        actualSeqLengthsQ = inputs[10]["value"] if len(inputs) > 10 else [query.shape[1]]
        actualSeqLengthsK = inputs[11]["value"] if len(inputs) > 11 else [key.shape[1]]
        scaleValue = inputs[12]["value"] if len(inputs) > 12 else 1.0 / math.sqrt(query.shape[-1])
        layout = inputs[13]["value"] if len(inputs) > 13 else "BSND"

        input_groups.append([
            query, key, queryIndex, keyIndex, weights, sparseIndices,
            softmaxMax, softmaxSum, queryRope, keyRope,
            actualSeqLengthsQ, actualSeqLengthsK, scaleValue, layout
        ])

    return input_groups


def get_init_inputs():
    return []
