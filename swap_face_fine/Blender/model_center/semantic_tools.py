import torch
from swap_face_fine.Blender.utils.project_kits import denorm_img
import torch.nn.functional as F
import sys


def _compute_fw_and_bw_sample_grid(score_map, point_num):
    B, H, W = score_map.size()
    # compute forward grid
    forward_grid = score_map.new_ones(B, point_num, 2) * -10
    point_scores = score_map.new_ones(B, point_num, 1) * -10

    score_map_f = score_map.reshape(B, H * W)
    point_probs_f, point_indices_f = torch.topk(score_map_f, k=point_num, dim=1)
    point_probs_per = point_probs_f.reshape(B, point_num)
    point_indices = point_indices_f.reshape(B, point_num)

    ws = (point_indices % W).to(torch.float) * 2 / (W - 1) - 1
    hs = (point_indices // W).to(torch.float) * 2 / (H - 1) - 1
    forward_grid[:, :, 0] = ws
    forward_grid[:, :, 1] = hs
    point_scores[:, :, 0] = point_probs_per
    assert forward_grid.min() >= -1
    assert forward_grid.max() <= 1

    # compute backward grid
    backward_grid = score_map.new_ones(B, H * W, 2) * -10
    for i in range(B):
        ws = torch.linspace(0, point_num - 1, point_num) * 2 / (point_num - 1) - 1
        hs = score_map.new_zeros(point_num)
        backward_grid[i, point_indices_f[i], 0] = ws.type_as(score_map)
        backward_grid[i, point_indices_f[i], 1] = hs

    return forward_grid[:, None], backward_grid.reshape(B, H, W, 2)


def chunk_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1):
    # E.g. x1 in (1,D,H,1), x2 in (1,D,1,W), then peak_memory=(1,D,H,W) during calculating (x1 @ x2)
    b, d, h, _ = x1.shape
    b, d, _, w = x2.shape
    h_split = h // 2
    x1_chunk1 = x1[:, :, :h_split, :]
    x1_chunk2 = x1[:, :, h_split:, :]
    res_chunk1 = F.cosine_similarity(x1_chunk1, x2, dim=dim)  # (1,D,h_split,W)
    res_chunk2 = F.cosine_similarity(x1_chunk2, x2, dim=dim)  # (1,D,h_split,W)
    res = torch.cat([res_chunk1, res_chunk2], dim=-2)
    return res


def get_color_refer(img_T, feats_A, feats_T, part_dict_A, part_dict_T, trainable_tao, compute_inv=True, light=False):
    r"""
    Args:
        img_T: (1,3,256,256), ori target
        feats_A: (1,256,64,64), FPN features of animated
        feats_T: (1,256,64,64), FPN features of target
        part_dict_A: {'head':(1,256,256);...}, in [0,1], mask dict of animated
        part_dict_T: {'head':(1,256,256);...}, in [0,1], mask dict of target
        trainable_tao: FloatTensor, softmax param
        compute_inv: bool
        light: bool, use smallFPN if True
    Returns:
        color_ref_dict: {'head':(1,)}
        color_inv_ref_pair:
    """
    color_ref_dict = {}
    color_inv_ref_dict = {}

    if light:
        operate_size = [50, 50]
        feats_A = F.upsample_nearest(feats_A, operate_size)
        feats_T = F.upsample_nearest(feats_T, operate_size)
    else:
        operate_size = feats_A.shape[-2:]  # [64,64]

    re_img_T = denorm_img(F.upsample_nearest(img_T, operate_size))

    # Keys:['skin', 'hair', 'eye', 'nose', 'lip', 'tooth', 'ear', 'brow', 'head', 'inpainting']
    for name in part_dict_A.keys():
        if name == 'head':
            continue

        mask_A = F.upsample_nearest(part_dict_A[name][:, None].float(), operate_size)[:, 0]  # (1,64,64)
        mask_T = F.upsample_nearest(part_dict_T[name][:, None].float(), operate_size)[:, 0]  # (1,64,64)
        N_As = mask_A.sum([1, 2])  # (1,), #pixels in 64x64 seg map, should < 4096=64*64
        N_Ts = mask_T.sum([1, 2])  # (1,), #pixels in 64x64 seg map, should < 4096=64*64

        if light:
            N_A = min(N_As.max().long().item(), 1000)
            N_T = min(N_Ts.max().long().item(), 1000)
        else:
            N_A = N_As.max().long().item()  # max #pixels of a facial part along batch_size
            N_T = N_Ts.max().long().item()  # max #pixels of a facial part along batch_size

        if min(N_A, N_T) == 0:
            continue

        fw_grid_A, bw_grid_A = _compute_fw_and_bw_sample_grid(mask_A, N_A)
        fw_grid_T, bw_grid_T = _compute_fw_and_bw_sample_grid(mask_T, N_T)

        flat_feat_A = F.grid_sample(feats_A * mask_A[:, None], fw_grid_A,
                                    mode='nearest', align_corners=True).squeeze(2)
        flat_feat_T = F.grid_sample(feats_T * mask_A[:, None], fw_grid_T,
                                    mode='nearest', align_corners=True).squeeze(2)
        flat_RGB_T = F.grid_sample(re_img_T, fw_grid_T,
                                   mode='nearest', align_corners=True).squeeze(2)  # (1,3,N_T)

        flat_feat_A = flat_feat_A - flat_feat_A.mean([1], keepdim=True)  # (1,256,N_A)
        flat_feat_T = flat_feat_T - flat_feat_T.mean([1], keepdim=True)  # (1,256,N_T)

        # Move to CPU for calculating cosine_similarity, to reduce GPU mem usage.
        # E.g. x1 in (1,D,H,1), x2 in (1,D,1,W), then peak_memory=(1,D,H,W) during calculating (x1 @ x2)
        max_matrix_size = 2000 * 2000
        use_fp16_tmp = flat_feat_A.shape[-1] * flat_feat_T.shape[-1] >= max_matrix_size
        # print("debug:", flat_feat_A.shape, flat_feat_T.shape)
        if use_fp16_tmp:
            # print("[Warning] exceed max matrix mem, calculating cos_sim in cpu.", flat_feat_A.shape, flat_feat_T.shape)
            # matrix = F.cosine_similarity(flat_feat_A.unsqueeze(-1).half(), flat_feat_T.unsqueeze(-2).half(), dim=1)
            # matrix = F.cosine_similarity(flat_feat_A.unsqueeze(-1).cpu(), flat_feat_T.unsqueeze(-2).cpu(), dim=1)
            print("[Warning] exceed max matrix mem, calculating cos_sim chunked in cpu.", flat_feat_A.shape, flat_feat_T.shape)
            matrix = chunk_cosine_similarity(flat_feat_A.unsqueeze(-1).cpu(), flat_feat_T.unsqueeze(-2).cpu(), dim=1)
            matrix = matrix.to(dtype=flat_feat_A.dtype)
        else:
            matrix = F.cosine_similarity(flat_feat_A.unsqueeze(-1), flat_feat_T.unsqueeze(-2), dim=1)
        matrix = matrix.to(flat_feat_A.device)
        att_matrix = F.softmax(matrix * trainable_tao, dim=-1)  # (1,N_A,N_T)

        # [B,N_A,3]=[B,N_A,N_T]*[B,N_T,3]
        flat_color_ref = torch.bmm(att_matrix, flat_RGB_T.permute(0, 2, 1))  # (1,N_A,3)

        color_ref = F.grid_sample(flat_color_ref.permute(0, 2, 1).unsqueeze(-2),
                                  bw_grid_A, mode='nearest', align_corners=True).squeeze(2)

        color_ref_dict[name] = color_ref * mask_A[:, None]  # recolored A in 'name' seg regions, others are black (zero)
        # print('[DEBUG] color_ref_dict:', color_ref_dict[name].shape)

        if compute_inv:
            inv_matrix = matrix.permute(0, 2, 1)
            inv_att_matrix = F.softmax(inv_matrix * trainable_tao, dim=-1)

            inv_flat_RGB_A = F.grid_sample(color_ref, fw_grid_A, mode='nearest', align_corners=True).squeeze(2)

            # [B,N_T,3]=[B,N_T,N_A]*[B,N_A,3]
            inv_flat_color_ref = torch.bmm(inv_att_matrix, inv_flat_RGB_A.permute(0, 2, 1))

            inv_color_ref = F.grid_sample(inv_flat_color_ref.permute(0, 2, 1).unsqueeze(-2),
                                          bw_grid_T, mode='nearest', align_corners=True).squeeze(2)

            color_inv_ref_dict[name] = inv_color_ref * mask_T[:, None]

            matrix

    if compute_inv:
        # color_inv_ref = sum([v for k, v in color_inv_ref_dict.items() if k != 'inpainting'])
        # color_inv_ref_pair = [color_inv_ref,
        #                       re_img_T * F.upsample_nearest(part_dict_T['head'][:, None].float(), operate_size)]

        color_inv_ref = sum([v for k, v in color_inv_ref_dict.items()])
        color_inv_ref_pair = [color_inv_ref,
                              re_img_T * F.upsample_nearest(
                                  (part_dict_T['head'] + part_dict_T['inpainting'])[:, None].float(),
                                  operate_size)]

        d = 1
    else:
        color_inv_ref_pair = []

    return color_ref_dict, color_inv_ref_pair


name_to_ids = {
    'skin': [1],
    'hair': [17],
    'eye': [4, 5],
    'nose': [10],
    'lip': [12, 13],
    'tooth': [11],
    'ear': [7, 8],
    'brow': [2, 3],
}


def get_part_dict(masks):
    part_dict = {}
    for name, ids in name_to_ids.items():
        part = sum([masks == id for id in ids])
        part_dict[name] = part
    part_dict['head'] = sum(list(part_dict.values()))
    return part_dict


def get_greyscale_head(img_A, mask_A_head):
    img01 = denorm_img(img_A)
    res = img01[:, 0] * 0.299 + img01[:, 1] * 0.587 + img01[:, 2] * 0.114
    return res.clamp(0, 1) * mask_A_head


def get_dilated_mask(mask, ratio=0.1):
    k = int(mask.shape[-1] * ratio / 2) * 2 + 1
    res = F.max_pool2d(mask[:, None].float(), kernel_size=k, stride=1, padding=k // 2)[:, 0].long()
    return res
