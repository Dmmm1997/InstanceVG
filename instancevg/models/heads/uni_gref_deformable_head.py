import torch
from torch import nn
from torch.nn import functional as F
import torch
from instancevg.core.criterion.criterion import SetCriterion
from instancevg.core.matcher import HungarianMatcher
from instancevg.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from instancevg.layers.mlp import MLP
from instancevg.layers.position_embedding import (
    PositionEmbeddingSine,
    PositionEmbeddingSine1D,
)
from instancevg.models import HEADS
import pycocotools.mask as maskUtils
import numpy as np

from instancevg.models.heads.tgqs_kd_detr_head.transformer import (
    DetrTransformer,
    DetrTransformerDecoder,
    DetrTransformerEncoder,
)
from instancevg.models.heads.transformers.deformable_transformer import (
    DeformableDetrTransformer,
    DeformableDetrTransformerDecoder,
    inverse_sigmoid,
)
from ..losses.contristiveloss import HardMiningTripletLoss
from ..losses import CEMLoss
from .modules import BoxSegAttention, BoxSegPooler, SegBranch, BoxBranch, QueryAugment
from .unet_head import SimpleFPN, SimpleDecoding, UnetDecoder
from ..losses.clip_loss import ClipLoss, get_rank, get_world_size
from .projection import Projector
from instancevg.models.losses.segloss import refer_ce_loss, part_seg_loss, seg_loss
from instancevg.utils.visual_utils import (
    visualize_attention_with_image,
    heatmap_visulization,
)


class TGQG_Simple(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_queries,
    ):
        super(TGQG_Simple, self).__init__()
        self.num_queries = num_queries
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_channels)
        self.text_guided_query_generation_transformer = DetrTransformerDecoder(
            embed_dim=hidden_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=512,
            ffn_dropout=0.1,
            num_layers=1,
            return_intermediate=False,
            post_norm=True,
        )

    def forward(self, text_feat, text_mask):
        query_embed_input = self.query_embed.weight.unsqueeze(0).repeat(text_feat.shape[0], 1, 1).transpose(0, 1)
        target = torch.zeros_like(query_embed_input)
        text_pos_embed = (
            self.position_embedding_1d(text_feat).unsqueeze(0).repeat(text_feat.shape[0], 1, 1).permute(1, 0, 2).cuda()
        )
        text_feat_input = text_feat.transpose(0, 1)
        query_embed, attn_map = self.text_guided_query_generation_transformer(
            query=target,
            key=text_feat_input,
            value=text_feat_input,
            key_pos=text_pos_embed,
            query_pos=query_embed_input,
            key_padding_mask=text_mask.bool(),
        )
        query_embed = query_embed[-1].transpose(0, 1)
        attn_map = attn_map[-1]
        return query_embed, attn_map


class TGQG(nn.Module):
    def __init__(
        self,
        hidden_channels=256,
        num_queries=10,
    ):
        super(TGQG, self).__init__()
        self.num_queries = num_queries
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_channels)
        self.text_guided_query_generation_transformer = DetrTransformerDecoder(
            embed_dim=hidden_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=512,
            ffn_dropout=0.1,
            num_layers=1,
            return_intermediate=False,
            post_norm=True,
        )

    def forward(self, cls_feat, text_feat, text_mask):
        cls_feat = cls_feat.unsqueeze(1).repeat((1, self.num_queries, 1))
        text_feat_filter = (
            torch.cat(
                list(
                    map(
                        lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0],
                        text_feat,
                        ~text_mask,
                    )
                )
            )
            .unsqueeze(1)
            .repeat(1, self.num_queries, 1)
        )
        query_embed_input = self.query_embed.weight.unsqueeze(0).repeat(cls_feat.shape[0], 1, 1).transpose(0, 1)
        target = torch.zeros_like(query_embed_input)
        text_pos_embed = (
            self.position_embedding_1d(text_feat).unsqueeze(0).repeat(text_feat.shape[0], 1, 1).permute(1, 0, 2).cuda()
        )
        text_feat_input = text_feat.transpose(0, 1)
        query_embed, attn_map = self.text_guided_query_generation_transformer(
            query=target,
            key=text_feat_input,
            value=text_feat_input,
            key_pos=text_pos_embed,
            query_pos=query_embed_input,
            key_padding_mask=text_mask.bool(),
        )
        query_embed = query_embed[0].transpose(0, 1) + text_feat_filter + query_embed_input.transpose(0, 1)
        cls_feat = query_embed + cls_feat
        return cls_feat


# Attention-guided Query Generation Module
class AQSM(nn.Module):
    def __init__(
        self,
        hidden_channels=256,
        num_queries=10,
        dist_guided_query_select_params={"enable": True, "W_dist": 0.1},
        score_text_select=False,
    ):
        super(AQSM, self).__init__()
        self.score_text_select = score_text_select
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_channels)
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.T2I_decoder = DetrTransformerDecoder(
            embed_dim=hidden_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=512,
            ffn_dropout=0.1,
            num_layers=1,
            return_intermediate=False,
            post_norm=True,
        )
        self.dist_guided_query_select_params = dist_guided_query_select_params
        self.mlp = MLP(hidden_channels * 2, hidden_channels, hidden_channels, 3)

    def x_mask_pos_enc(self, x, img_shape):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_shape
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)
        x_pos_embeds = self.position_embedding(x_mask)
        return x_mask, x_pos_embeds

    def text_pooler(self, lan_feat, lan_mask):
        lan_feat_pooler = torch.cat(
            list(
                map(
                    lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0],
                    lan_feat,
                    lan_mask.bool(),
                )
            )
        )
        return lan_feat_pooler

    def text_selector(self, lan_feat, lan_mask, select_num=10):
        # lan_feat: shape (B, 50, C)
        # lan_mask: shape (B, 50)
        B, L, C = lan_feat.shape
        # Step 1: 通过 lan_mask 选出有效的 lan_feat
        lan_mask_expanded = lan_mask.unsqueeze(-1).expand(-1, -1, C)  # Shape: (B, 50, C)
        # 只保留有效位置的特征
        valid_lan_feat = lan_feat * lan_mask_expanded  # Shape: (B, 50, C)
        # 计算每个特征向量的 L2 范数作为分数
        # scores = valid_lan_feat.norm(dim=-1)  # Shape: (B, 50)
        scores = valid_lan_feat.mean(dim=-1)
        # 计算每个 batch中有效特征的数量
        valid_counts = lan_mask.sum(dim=1)  # Shape: (B,)
        # 初始化用于存储选择的特征和对应的 mask 的 tensor
        selected_lan_feat = torch.zeros((B, select_num, C), dtype=lan_feat.dtype, device=lan_feat.device)
        selected_mask = torch.zeros((B, select_num), dtype=lan_mask.dtype, device=lan_mask.device)
        # Step 2: 通过 topk 选出 select_num 个 lan_feat
        for i in range(B):
            valid_num = valid_counts[i].item()
            if valid_num >= select_num:
                # 如果有效特征数大于等于 select_num，则直接选择 topk
                _, topk_indices = torch.topk(scores[i, :valid_num], select_num)
                selected_lan_feat[i] = valid_lan_feat[i, topk_indices]
                selected_mask[i] = 1  # 所有选择的特征都是有效的
            else:
                # 否则选择所有有效特征，并重复补足 select_num
                selected_lan_feat[i] = valid_lan_feat[i, :select_num]
                selected_mask[i, :valid_num] = 1  # 标记有效特征
        return selected_lan_feat, selected_mask

    def point_selector(self, matrix, num_points, W_dist=0.1):
        matrix = matrix.sigmoid()
        batchsize, h, w = matrix.shape
        # 创建所有点的列表 (i, j)
        points = torch.tensor([(i, j) for i in range(h) for j in range(w)], dtype=torch.float32).to(matrix.device)
        # 初始化结果列表
        results = torch.zeros(batchsize, num_points, dtype=torch.int64, device=matrix.device)
        # 1. 首先选择每个 batch 中得分最高的点
        max_score_indices = torch.argmax(matrix.view(batchsize, -1), dim=1)
        max_points = torch.stack((max_score_indices // w, max_score_indices % w), dim=1).float()  # (batchsize, 2)
        # 将选出的点加入结果中，并从 points 中移除
        results[:, 0] = max_score_indices
        added_points = max_points.unsqueeze(1)  # (batchsize, 1, 2)
        # 剩余的点
        remaining_points = points.unsqueeze(0).repeat(batchsize, 1, 1)  # (batchsize, num_points, 2)
        mask = torch.ones(batchsize, h * w, dtype=torch.bool, device=matrix.device)
        mask.scatter_(1, max_score_indices.unsqueeze(1), False)
        # 更新 remaining_points，移除已选择的点
        mask_reshaped = mask.view(batchsize, h, w)
        remaining_points = (
            torch.stack(
                torch.meshgrid(torch.arange(h, device=matrix.device), torch.arange(w, device=matrix.device)),
                dim=-1,
            )
            .view(-1, 2)
            .unsqueeze(0)
            .repeat(batchsize, 1, 1)
        )
        remaining_points = remaining_points[mask_reshaped.view(batchsize, -1)].view(batchsize, -1, 2)
        # 2. 迭代选择其余点
        for i in range(1, num_points):
            # 计算所有剩余点到已选点的最小距离
            dists = (
                torch.cdist(remaining_points.float(), added_points).min(dim=2).values
            )  # (batchsize, num_remaining_points)
            # 综合考虑得分和距离，计算每个点的综合评分
            scores = matrix.view(batchsize, -1).masked_select(mask).view(batchsize, -1)
            combined_scores = scores + W_dist * dists / np.sqrt(h * h + w * w)
            # 找到综合得分最高的点
            best_idx = combined_scores.argmax(dim=1)
            best_point = remaining_points[torch.arange(batchsize), best_idx]  # (batchsize, 2)
            # 更新结果
            flat_indices = mask.nonzero(as_tuple=False).view(batchsize, -1, 2)
            results[:, i] = flat_indices[torch.arange(batchsize), best_idx, 1]  # 获取每个 batch 中的最佳点索引
            # results[:, i] = mask.nonzero(as_tuple=False).view(batchsize, h * w)[torch.arange(batchsize), best_idx]
            # 将选出的点加入 added_points 中
            added_points = torch.cat((added_points, best_point.unsqueeze(1)), dim=1)
            # 更新 mask，移除已选点
            mask.scatter_(1, results[:, i].unsqueeze(1), False)
            # 更新 remaining_points，移除已选择的点
            mask_reshaped = mask.view(batchsize, h, w)
            remaining_points = (
                torch.stack(
                    torch.meshgrid(torch.arange(h, device=matrix.device), torch.arange(w, device=matrix.device)),
                    dim=-1,
                )
                .view(-1, 2)
                .unsqueeze(0)
                .repeat(batchsize, 1, 1)
            )
            remaining_points = remaining_points[mask_reshaped.view(batchsize, -1)].view(batchsize, -1, 2)
        return results

    def forward(self, text_feat, text_mask, img_feat):
        b, c, h, w = img_feat.shape
        # *
        if self.score_text_select:
            selected_text_feat, selected_mask = self.text_selector(
                text_feat, 1 - text_mask, select_num=self.num_queries
            )
        else:
            # selected_text_feat = text_feat[:, :10]
            # selected_mask = text_mask[:, :10]
            selected_text_feat = torch.topk(text_feat, self.num_queries, dim=1)[0]
            selected_mask = torch.ones_like(text_mask[:, : self.num_queries])
        text_feat_input = selected_text_feat.transpose(0, 1)
        target = torch.zeros_like(text_feat_input)
        img_masks, pos_embed = self.x_mask_pos_enc(img_feat, img_feat.shape[-2:])
        img_feat_input = img_feat.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed, attn_map = self.T2I_decoder(
            query=target,
            key=img_feat_input,
            value=img_feat_input,
            key_pos=pos_embed,
            query_pos=text_feat_input,
            key_padding_mask=img_masks.flatten(1).bool(),
        )
        text_feat = query_embed[-1].transpose(0, 1)
        # filterd_topk_textfeat, filtered_topk_inds = torch.topk(text_feat, self.num_queries, dim=1)
        global_attn_map = self.text_pooler(attn_map[-1], selected_mask)
        if not self.dist_guided_query_select_params["enable"]:
            pos_inds = torch.topk(global_attn_map, self.num_queries, dim=-1)[1]
        else:
            pos_inds = self.point_selector(
                global_attn_map.reshape(b, h, w),
                self.num_queries,
                self.dist_guided_query_select_params["W_dist"],
            )
        img_feat_ = img_feat.flatten(2).transpose(1, 2)  #  (B, HxW, C)
        pos_inds_ = pos_inds.flatten(1).unsqueeze(-1).expand(-1, -1, c)  #  (B, 50 * 5)
        pos_feat = torch.gather(img_feat_, 1, pos_inds_)
        pos_points = torch.stack((((pos_inds % w).float() + 0.5) / w, ((pos_inds // w).float() + 0.5) / h), dim=-1)
        pos_feat = self.mlp(torch.concat((text_feat, pos_feat), dim=-1))

        return (
            pos_feat,
            pos_points,
            global_attn_map.reshape(b, h, w),
            attn_map[-1].reshape(b, self.num_queries, h, w),
        )


#
class TFQG(nn.Module):
    def __init__(
        self,
        hidden_channels=256,
        num_queries=10,
    ):
        super(TFQG, self).__init__()
        self.num_queries = num_queries

    def forward(self, text_feat, text_mask, img_feat):
        b, c, h, w = img_feat.shape
        # text_feat_filter = text_feat[text_mask.bool(), :]
        filterd_topk, filtered_topk_inds = torch.topk(text_feat, self.num_queries, dim=1)
        return filterd_topk


class DETRLoss(nn.Module):
    def __init__(
        self,
        criterion={"loss_class": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
        matcher={"cost_class": 1.0, "cost_bbox": 5.0, "cost_giou": 2.0, "cost_point": 3.0},
        aux_loss=True,
        num_classes=1,
    ):
        super(DETRLoss, self).__init__()
        self.aux_loss = aux_loss
        self.matcher = HungarianMatcher(
            cost_class=matcher["cost_class"],
            cost_bbox=matcher["cost_bbox"],
            cost_giou=matcher["cost_giou"],
            cost_point=matcher["cost_point"],
            cost_class_type="ce_cost",
        )
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict={
                "loss_class": criterion["loss_class"],
                "loss_bbox": criterion["loss_bbox"],
                "loss_giou": criterion["loss_giou"],
            },
            loss_class_type="ce_loss",
            eos_coef=0.1,
        )

    def _set_aux_loss(self, outputs_class, outputs_coord, pred_selected_points):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_selected_points": pred_selected_points,
            }
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def forward(self, output_class, output_coord, targets, pred_selected_points):
        output = {
            "pred_logits": output_class[-1],
            "pred_boxes": output_coord[-1],
            "pred_selected_points": pred_selected_points,
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(output_class, output_coord, pred_selected_points)
        loss_dict, indices = self.criterion(output, targets, return_indices=True)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict, indices


@HEADS.register_module()
class UniGRefDeformableHead(nn.Module):
    def __init__(
        self,
        input_channels=768,
        hidden_channels=256,
        loss_weight={"mask": 1.0, "bbox": 0.025, "cons": 0.0},
        num_queries=10,
        detr_loss={},
        query_generation_type="tgqg",
        score_text_select=False,
        dist_guided_query_select_params=None,
    ):
        super(UniGRefDeformableHead, self).__init__()
        self.seg_branch = SegBranch(hidden_channels, upsample_rate=1)
        if "aux" in loss_weight["mask"]:
            self.seg_branch_aux_list = nn.ModuleList(
                [SegBranch(hidden_channels // 4 * (2**i), upsample_rate=1) for i in range(4)]
            )
        self.nt_embed = MLP(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=2,
            num_layers=3,
        )

        self.class_embed = nn.Linear(hidden_channels, 1 + 1)
        self.bbox_embed = MLP(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=4,
            num_layers=3,
        )

        self.loss_weight = loss_weight

        self.lan_embedding = nn.Linear(input_channels, hidden_channels, bias=False)
        self.img_embedding = nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=False)
        self.query_embedding = nn.Linear(input_channels, hidden_channels, bias=False)

        self.neck = SimpleFPN(
            backbone_channel=hidden_channels,
            in_channels=[
                hidden_channels // 4,
                hidden_channels // 2,
                hidden_channels,
                hidden_channels,
            ],
            out_channels=[
                hidden_channels,
                hidden_channels,
                hidden_channels,
                hidden_channels,
            ],
        )
        self.unet_decoder = UnetDecoder(hidden_channels, 1)

        self.proj_pixel_level_cons = Projector(
            word_dim=hidden_channels,
            in_dim=hidden_channels,
            hidden_dim=hidden_channels // 2,
            kernel_size=1,
        )
        self.query_generation_type = query_generation_type
        if query_generation_type == "tgqg":
            self.query_generation = TGQG(hidden_channels=hidden_channels, num_queries=num_queries)
        elif query_generation_type == "aqsm":
            self.query_generation = AQSM(
                hidden_channels=hidden_channels,
                num_queries=num_queries,
                score_text_select=score_text_select,
                dist_guided_query_select_params=dist_guided_query_select_params,
            )
        elif query_generation_type == "tfqg":
            self.query_generation = TFQG(hidden_channels=hidden_channels)
        elif query_generation_type == "tgqg_simple":
            self.query_generation = TGQG_Simple(hidden_channels=hidden_channels, num_queries=num_queries)
        else:
            raise NotImplementedError("query_generation_type not implemented")
        self.detrloss = DETRLoss(
            aux_loss=True,
            criterion=detr_loss["criterion"],
            matcher=detr_loss["matcher"],
            num_classes=1,
        )
        self.decoder_transformer = DeformableDetrTransformer(
            encoder=None,
            decoder=DeformableDetrTransformerDecoder(
                embed_dim=256,
                num_heads=8,
                feedforward_dim=1024,
                attn_dropout=0.1,
                ffn_dropout=0.1,
                num_layers=3,
                return_intermediate=True,
            ),
            only_decoder=True,
        )
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        num_pred = self.decoder_transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

        self.query_pos_embedding = nn.Linear(2, hidden_channels)
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )

    def prepare_targets(self, targets, mask_parts, img_metas):
        new_targets = []
        is_empty = []
        for target_bbox, mask_part, img_meta in zip(targets, mask_parts, img_metas):
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if img_meta["empty"]:
                target_bbox = torch.zeros((0, 4), device=target_bbox.device)
            else:
                target_bbox = target_bbox.reshape(-1, 4)
            gt_classes = torch.zeros(target_bbox.shape[0], device=target_bbox.device).long()
            is_empty.append(int(img_meta["empty"]))
            gt_boxes = target_bbox.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "masks": mask_part})
        is_empty = torch.tensor(is_empty, device=target_bbox.device).long()
        return new_targets, is_empty

    def text_pooler(self, lan_feat, lan_mask):
        lan_feat_pooler = torch.cat(
            list(
                map(
                    lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0],
                    lan_feat,
                    ~lan_mask,
                )
            )
        )
        return lan_feat_pooler

    def x_mask_pos_enc(self, x, img_shape):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_shape
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)
        x_pos_embeds = self.position_embedding(x_mask)
        return x_mask, x_pos_embeds

    def forward_train(self, x, targets, cls_feat=None, lan_feat=None, lan_mask=None, img=None):
        device = x.device
        extra_dict = {}
        target_mask = torch.from_numpy(
            np.concatenate([maskUtils.decode(target)[None] for target in targets["mask"]])
        ).to(device)
        target_mask_parts = []
        for mask_part in targets["mask_parts"]:
            if not mask_part:
                mask_list = np.zeros((0, img.shape[2], img.shape[3]), dtype=np.uint8)
            else:
                mask_list = np.concatenate([maskUtils.decode(single_mask)[None] for single_mask in mask_part])
            target_mask_parts.append(mask_list)
        img_metas = targets["img_metas"]
        # all feats embedding to hidden_channels
        img_feat = self.img_embedding(x)
        query_feat = self.query_embedding(cls_feat)
        lan_feat = self.lan_embedding(lan_feat)

        # ! neck
        x_c1, x_c2, x_c3, x_c4 = self.neck(img_feat)
        multi_level_feats = [x_c1, x_c2, x_c3, x_c4]
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in [x_c1, x_c2, x_c3, x_c4]:
            img_masks, pos_embed = self.x_mask_pos_enc(feat, feat.shape[-2:])
            multi_level_masks.append(img_masks.to(torch.bool))
            multi_level_position_embeddings.append(pos_embed)

        reference_point = None
        # ! query generation
        if self.query_generation_type == "tgqg":
            query_feat = self.query_generation(query_feat, lan_feat, lan_mask.bool())
        elif self.query_generation_type == "aqsm":
            query_feat, reference_point, attn_map_query, attn_map = self.query_generation(lan_feat, lan_mask, img_feat)
            extra_dict["img_selected_points"] = reference_point
            extra_dict["attn_map_query"] = attn_map_query
            extra_dict["attn_map"] = attn_map
        elif self.query_generation_type == "tfqg":
            query_feat = self.query_generation(lan_feat, lan_mask.bool(), img_feat)
        elif self.query_generation_type == "tgqg_simple":
            query_feat, q2t_attn_map = self.query_generation(lan_feat, lan_mask.bool())
            extra_dict["q2t_attn_map"] = q2t_attn_map

        # ! decoder
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.decoder_transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_feat,
            self.query_pos_embedding(reference_point),
            reference_point,
        )  # (B, N, C)
        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # ! segmentaton branch
        pred_mask = self.unet_decoder(x_c4, x_c3, x_c2, x_c1)  # (B, C, H, W)
        all_mask = torch.einsum("bqc,bchw->bqhw", inter_states[-1], pred_mask)  # (B, N, H, W)
        pred_seg = self.seg_branch(pred_mask)  # (B, 1, H, W)
        nt_label = self.nt_embed(pred_mask.flatten(2).transpose(1, 2)).mean(1)  # (B, 2)

        all_mask = F.interpolate(all_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)
        pred_seg = F.interpolate(pred_seg, size=img.shape[-2:], mode="bilinear", align_corners=True)

        # prepare the target fot detection
        target_gt, is_empty = self.prepare_targets(targets["bbox"], target_mask_parts, img_metas)
        # ! detr loss
        det_loss_dict, indices = self.detrloss(outputs_class, outputs_coord, target_gt, reference_point)
        loss_det = sum(det_loss_dict.values()) * self.loss_weight["bbox"]
        # ! part seg loss
        loss_part_mask = part_seg_loss(all_mask, target_mask_parts, indices, self.loss_weight["mask"])
        # ! seg loss
        loss_mask = seg_loss(pred_seg, target_mask, self.loss_weight["mask"])
        loss_mask_nt = (
            refer_ce_loss(nt_label, is_empty, torch.FloatTensor([0.9, 1.1]).to(nt_label))
            * self.loss_weight["mask"]["nt"]
        )
        # ! coarse-stage seg loss
        attn_map_target = F.interpolate(target_mask.unsqueeze(1), size=attn_map_query.shape[-2:], mode="nearest")
        loss_attnmapmask = (
            seg_loss(attn_map_query.unsqueeze(1), attn_map_target.squeeze(1), self.loss_weight["mask"]) * 0.1
        )
        # ! aux loss
        aux_loss = torch.zeros(1, device=loss_mask.device)
        if "aux" in self.loss_weight["mask"]:
            for ind, x_ in enumerate([x_c1, x_c2, x_c3, x_c4]):
                x_ = self.seg_branch_aux_list[ind](x_)
                x_pred = F.interpolate(x_, size=img.shape[-2:], mode="bilinear", align_corners=True)
                aux_loss += seg_loss(x_pred, target_mask, self.loss_weight["mask"]) * self.loss_weight["mask"]["aux"]

        loss_dict = {
            "loss_mask": loss_part_mask + loss_mask + aux_loss + loss_attnmapmask,
            "loss_det": loss_det,
            "loss_label_nt": loss_mask_nt,
        }
        pred_dict = {
            "pred_mask": all_mask.detach(),
            "pred_global_mask": pred_seg.detach(),
            "pred_class": outputs_class[-1].detach(),
            "pred_bbox": outputs_coord[-1].detach(),
            "nt_label": nt_label.detach(),
        }
        return loss_dict, pred_dict, extra_dict

    def forward_test(self, x, cls_feat=None, lan_feat=None, lan_mask=None, img=None, targets=None):
        # all feats embedding to hidden_channels
        extra_dict = {}
        img_feat = self.img_embedding(x)
        query_feat = self.query_embedding(cls_feat)
        lan_feat = self.lan_embedding(lan_feat)

        # ! neck
        x_c1, x_c2, x_c3, x_c4 = self.neck(img_feat)
        multi_level_feats = [x_c1, x_c2, x_c3, x_c4]
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in [x_c1, x_c2, x_c3, x_c4]:
            img_masks, pos_embed = self.x_mask_pos_enc(feat, feat.shape[-2:])
            multi_level_masks.append(img_masks.to(torch.bool))
            multi_level_position_embeddings.append(pos_embed)

        reference_point = None
        # ! query generation
        if self.query_generation_type == "tgqg":
            query_feat = self.query_generation(query_feat, lan_feat, lan_mask.bool())
        elif self.query_generation_type == "aqsm":
            query_feat, reference_point, attn_map_query, attn_map = self.query_generation(lan_feat, lan_mask, img_feat)
            extra_dict["img_selected_points"] = reference_point
            extra_dict["attn_map_query"] = attn_map_query
            extra_dict["attn_map"] = attn_map
        elif self.query_generation_type == "tfqg":
            query_feat = self.query_generation(lan_feat, lan_mask.bool(), img_feat)
        elif self.query_generation_type == "tgqg_simple":
            query_feat, q2t_attn_map = self.query_generation(lan_feat, lan_mask.bool())
            extra_dict["q2t_attn_map"] = q2t_attn_map

        # ! decoder
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.decoder_transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_feat,
            self.query_pos_embedding(reference_point),
            reference_point,
        )  # (B, N, C)
        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # ! segmentaton branch
        pred_mask = self.unet_decoder(x_c4, x_c3, x_c2, x_c1)  # (B, C, H, W)
        all_mask = torch.einsum("bqc,bchw->bqhw", inter_states[-1], pred_mask)  # (B, N, H, W)
        pred_seg = self.seg_branch(pred_mask)  # (B, 1, H, W)
        nt_label = self.nt_embed(pred_mask.flatten(2).transpose(1, 2)).mean(1)  # (B, 2)

        all_mask = F.interpolate(all_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)
        pred_seg = F.interpolate(pred_seg, size=img.shape[-2:], mode="bilinear", align_corners=True)

        pred_dict = {
            "pred_mask": all_mask.detach(),
            "pred_global_mask": pred_seg.detach(),
            "pred_class": outputs_class[-1].detach(),
            "pred_bbox": outputs_coord[-1].detach(),
            "nt_label": nt_label.detach(),
        }

        return pred_dict, extra_dict
