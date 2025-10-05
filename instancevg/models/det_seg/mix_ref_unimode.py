import copy
from torch.nn import functional as F
import torch
import numpy
from instancevg.core.structure.boxes import Boxes
from instancevg.core.structure.instances import Instances
from instancevg.core.structure.postprocessing import detector_postprocess
from instancevg.layers.box_ops import box_cxcywh_to_xyxy
from instancevg.models import MODELS
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils

from instancevg.models.postprocess.nms import nms
from .one_stage import OneStageModel
import numpy as np
from PIL import Image, ImageDraw
from instancevg.utils import is_main
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from copy import deepcopy


@MODELS.register_module()
class MIXRefUniModel(OneStageModel):
    def __init__(
        self,
        word_emb,
        num_token,
        vis_enc,
        lan_enc,
        head,
        fusion,
        mask_save_target_dir="",
        process_visual=True,
        post_params={
            "score_weighted": True,
            "mask_threshold": 0.5,
            "score_threshold": 0.7,
            "with_nms": False,
            "outmask_type": "merge",
        },
        visualize_params={"row_columns": (2, 5)},
        visual_mode="val",
    ):
        super(MIXRefUniModel, self).__init__(word_emb, num_token, vis_enc, lan_enc, head, fusion)
        self.patch_size = vis_enc["patch_size"]
        self.visualize = process_visual
        if is_main() and self.visualize:
            self.train_mask_save_target_dir = os.path.join(mask_save_target_dir, "train_vis")
            self.val_mask_save_target_dir = os.path.join(mask_save_target_dir, "val_vis")
            self.test_mask_save_target_dir = os.path.join(mask_save_target_dir, "test_vis")
            os.makedirs(self.train_mask_save_target_dir, exist_ok=True)
            os.makedirs(self.val_mask_save_target_dir, exist_ok=True)
            os.makedirs(self.test_mask_save_target_dir, exist_ok=True)
        self.iter = 0
        self.threshold = post_params["mask_threshold"]
        self.box_threshold = post_params["score_threshold"]
        self.score_weighted = post_params["score_weighted"]
        self.outmask_type = post_params["outmask_type"]
        self.with_nms = post_params["with_nms"]
        self.visualize_params = visualize_params
        self.visual_mode = visual_mode

    def forward_train(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        gt_bbox=None,
        gt_mask_rle=None,
        gt_mask_parts_rle=None,
        rescale=False,
        epoch=None,
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token].

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `seqtr/datasets/pipelines/formatting.py:CollectData`.

        gt_bbox (list[tensor]): [4, ], in [tl_x, tl_y, br_x, br_y] format,
            the coordinates are in 'img_shape' scale.

        gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1,
            the coordinates are in 'pad_shape' scale.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.

        """
        B, _, H, W = img.shape
        img_feat, text_feat, cls_feat = self.extract_visual_language(img, ref_expr_inds, text_attention_mask)
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)  # (B, C, H, W)

        targets = {
            "mask": gt_mask_rle,
            "bbox": gt_bbox,
            "img_metas": img_metas,
            "epoch": epoch,
            "mask_parts": gt_mask_parts_rle,
        }

        losses_dict, pred_dict, extra_dict = self.head.forward_train(
            img_feat, targets, cls_feat, text_feat, text_attention_mask, img
        )

        with torch.no_grad():
            predictions = self.get_predictions_parts(
                pred_dict, img_metas, rescale=rescale, with_bbox=True, with_mask=True
            )
        self.iter += 1
        # if is_main() and self.iter % 20 == 0 and self.visualize:
        #     self.visualiation_parts(
        #         predictions["parts_list"],
        #         img_metas,
        #         targets,
        #         self.train_mask_save_target_dir,
        #         extra_dict,
        #     )

        return losses_dict, _

    def extract_visual_language(self, img, ref_expr_inds, text_attention_mask=None):
        x, y, c = self.vis_enc(img, ref_expr_inds, text_attention_mask)
        return x, y, c

    @torch.no_grad()
    def forward_test(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        with_bbox=False,
        with_mask=False,
        gt_bbox=None,
        gt_mask=None,
        gt_mask_parts_rle=None,
        rescale=False,
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token], padded value is 0.

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `rec/datasets/pipelines/formatting.py:CollectData`.

        with_bbox/with_mask: whether to generate bbox coordinates or mask contour vertices,
            which has slight differences.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """
        self.iter += 1

        B, _, H, W = img.shape
        img_feat, text_feat, cls_feat = self.extract_visual_language(img, ref_expr_inds, text_attention_mask)
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)

        targets = {
            "mask": gt_mask,
            "bbox": gt_bbox,
            "img_metas": img_metas,
            "mask_parts": gt_mask_parts_rle,
        }

        pred_dict, extra_dict = self.head.forward_test(img_feat, cls_feat, text_feat, text_attention_mask, img)

        predictions = self.get_predictions_parts(
            pred_dict,
            img_metas,
            rescale=rescale,
            with_bbox=with_bbox,
            with_mask=with_mask,
        )

        # if is_main() and self.iter % 5 == 0 and self.visualize:
        #     self.visualiation_parts(
        #         predictions["parts_list"],
        #         img_metas,
        #         targets,
        #         self.val_mask_save_target_dir if self.visual_mode == "val" else self.test_mask_save_target_dir,
        #         extra_dict,
        #     )

        bs_acc_list = self.cover_acc(extra_dict["img_selected_points"], gt_bbox, img_metas)
        predictions["bs_acc_list"] = bs_acc_list
        return predictions

    def cover_acc(self, points, gt_bbox, img_metas):
        if gt_bbox is None:
            return None
        bs = len(gt_bbox)
        bs_acc_list = []
        for i in range(bs):
            H, W = img_metas[i]["batch_input_shape"]
            x_points, y_points = points[i][:, 0] * H, points[i][:, 1] * W
            # 初始化有效 target 数量
            valid_target_count = 0
            # 遍历每个 target
            for target in gt_bbox:
                x1, y1, x2, y2 = target
                # 判断哪些点落在当前 target 的内部
                in_x_range = (x_points >= x1) & (x_points <= x2)
                in_y_range = (y_points >= y1) & (y_points <= y2)
                # 如果有至少一个点在 target 内部，则该 target 有效
                if torch.any(in_x_range & in_y_range):
                    valid_target_count += 1

            # 计算有效 target 的比例
            total_targets = len(gt_bbox)
            if total_targets == 0:
                continue
            valid_target_ratio = valid_target_count / total_targets
            bs_acc_list.append(valid_target_ratio)
        return bs_acc_list

    def get_predictions(self, pred, img_metas, rescale=False, with_bbox=False, with_mask=False):
        """Args:
        seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """

        pred_bboxes, pred_masks = [], []
        bboxes, mask_seg, bbox_cls, nt_labels = (
            pred.get("pred_bbox", None),
            pred.get("pred_mask", None),
            pred.get("pred_class", None),
            pred.get("nt_label", None),
        )
        if bboxes is not None and with_bbox:
            image_sizes = [img_meta["img_shape"] for img_meta in img_metas]
            results = self.inference(bbox_cls, bboxes, image_sizes)
            for ind, (results_per_image, img_meta) in enumerate(zip(results, img_metas)):
                image_size = img_meta["img_shape"]
                height = image_size[0]
                width = image_size[1]
                r = detector_postprocess(results_per_image, height, width)
                # infomation extract
                pred_box = r.pred_boxes.tensor
                score = r.scores
                pred_class = r.pred_classes
                if rescale:
                    scale_factors = img_meta["scale_factor"]
                    pred_box /= pred_box.new_tensor(scale_factors)
                cur_predict_dict = {
                    "boxes": pred_box,
                    "scores": score,
                    "labels": pred_class,
                }
                pred_bboxes.append(cur_predict_dict)
        if mask_seg is not None and with_mask:
            mask_binary = mask_seg.sigmoid().squeeze(1)
            nt_labels = nt_labels.sigmoid()
            mask_binary[mask_binary < self.threshold] = 0.0
            mask_binary[mask_binary >= self.threshold] = 1.0
            for ind, (mask, img_meta) in enumerate(zip(mask_binary, img_metas)):
                h_pad, w_pad = img_meta["pad_shape"][:2]
                # h, w = img_meta['img_shape'][:2]
                pred_rle = maskUtils.encode(numpy.asfortranarray(mask.cpu().numpy().astype(np.uint8)))
                if rescale:
                    h_img, w_img = img_meta["ori_shape"][:2]
                    pred_mask = BitmapMasks(maskUtils.decode(pred_rle)[None], h_pad, w_pad)
                    pred_mask = pred_mask.resize((h_img, w_img))
                    pred_mask = pred_mask.masks[0]
                    pred_mask = numpy.asfortranarray(pred_mask)
                    pred_rle = maskUtils.encode(pred_mask)  # dict
                pred_nt = nt_labels[ind]
                gt_nt = img_meta["empty"]
                pred_masks.append({"pred_masks": pred_rle, "pred_nt": pred_nt, "gt_nt": gt_nt})

        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks)

    def get_predictions_parts(self, pred, img_metas, rescale=False, with_bbox=False, with_mask=False):
        """Args:
        seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """

        pred_bboxes, pred_masks = [], []
        bboxes, mask_seg, bbox_cls, nt_labels, global_seg_mask = (
            pred.get("pred_bbox", None),
            pred.get("pred_mask", None),
            pred.get("pred_class", None),
            pred.get("nt_label", None),
            pred.get("pred_global_mask", None),
        )
        scores, nms_indices = [], []
        nt_labels = nt_labels.sigmoid()
        # nt_labels = nt_labels.softmax(-1)
        parts_list = {
            "pred_mask_parts": [],
            "pred_box_parts": [],
            "pred_scores": [],
            "pred_mask": [],
        }
        if bboxes is not None and with_bbox:
            image_sizes = [img_meta["img_shape"] for img_meta in img_metas]
            results = self.inference(bbox_cls, bboxes, image_sizes)
            for ind, (results_per_image, img_meta) in enumerate(zip(results, img_metas)):
                pred_nt = nt_labels[ind]
                image_size = img_meta["img_shape"]
                height = image_size[0]
                width = image_size[1]
                r = detector_postprocess(results_per_image, height, width)
                # infomation extract
                pred_box = r.pred_boxes.tensor
                score = r.scores
                if self.score_weighted:
                    score = score * pred_nt[0]
                if self.with_nms:
                    filtered_boxes = copy.deepcopy(pred_box)
                    filtered_scores = copy.deepcopy(score)
                    filtered_indices = nms(filtered_boxes, filtered_scores, 0.7)
                    filtered_boxes = filtered_boxes[filtered_indices]
                    filtered_scores = filtered_scores[filtered_indices]
                    nms_indices.append(filtered_indices)
                if rescale:
                    scale_factors = img_meta["scale_factor"]
                    pred_box /= pred_box.new_tensor(scale_factors)
                if self.with_nms:
                    cur_predict_dict = {
                        "boxes": pred_box,
                        "scores": score,
                        "filtered_boxes": filtered_boxes,
                        "filtered_scores": filtered_scores,
                    }
                else:
                    cur_predict_dict = {"boxes": pred_box, "scores": score}
                parts_list["pred_scores"].append(score.cpu().detach().numpy())
                parts_list["pred_box_parts"].append(pred_box.cpu().detach().numpy())
                scores.append(score)
                pred_bboxes.append(cur_predict_dict)
        if mask_seg is not None and with_mask:

            mask_binary = mask_seg.sigmoid()  # (B, 10, H, W)
            mask_binary[mask_binary < self.threshold] = 0.0
            mask_binary[mask_binary >= self.threshold] = 1.0

            global_mask_binary = global_seg_mask.sigmoid()
            global_mask_binary[global_mask_binary < self.threshold] = 0.0
            global_mask_binary[global_mask_binary >= self.threshold] = 1.0
            for ind, (mask, img_meta, global_mask) in enumerate(zip(mask_binary, img_metas, global_mask_binary)):
                pred_nt = nt_labels[ind]
                h_pad, w_pad = img_meta["pad_shape"][:2]
                cur_scores = scores[ind]
                mask_tmp = copy.deepcopy(mask)
                if self.with_nms:
                    mask_tmp = mask_tmp[nms_indices[ind]]
                    cur_scores = cur_scores[nms_indices[ind]]
                indices = torch.where(cur_scores > self.box_threshold)
                mask_valid = mask_tmp[indices]

                if self.outmask_type == "global":
                    mask_ = torch.any(global_mask, dim=0).int()
                elif self.outmask_type == "instance":
                    mask_ = torch.any(mask_valid, dim=0).int()
                elif self.outmask_type == "merge":
                    mask_ = torch.any(torch.concat((global_mask, mask_valid), dim=0), dim=0).int()
                pred_rle = maskUtils.encode(numpy.asfortranarray(mask_.cpu().numpy().astype(np.uint8)))
                if rescale:
                    h_img, w_img = img_meta["ori_shape"][:2]
                    pred_mask = BitmapMasks(maskUtils.decode(pred_rle)[None], h_pad, w_pad)
                    pred_mask = pred_mask.resize((h_img, w_img))
                    pred_mask = pred_mask.masks[0]
                    pred_mask = numpy.asfortranarray(pred_mask)
                    pred_rle = maskUtils.encode(pred_mask)  # dict
                    mask_tmp = []
                    for m in mask:
                        m = BitmapMasks(m[None].cpu().detach().numpy(), h_pad, w_pad).resize((h_img, w_img))
                        m = numpy.asfortranarray(m.masks[0])
                        mask_tmp.append(m)
                    mask = np.stack(mask_tmp, axis=0)
                gt_nt = img_meta["empty"]
                pred_masks.append({"pred_masks": pred_rle, "pred_nt": pred_nt, "gt_nt": gt_nt})
                if rescale:
                    parts_list["pred_mask_parts"].append(mask)
                else:
                    parts_list["pred_mask_parts"].append(mask.cpu().detach().numpy())
                parts_list["pred_mask"].append(mask_.cpu().detach().numpy())

        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks, parts_list=parts_list)

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (
            scores_per_image,
            labels_per_image,
            box_pred_per_image,
            image_size,
        ) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def visualiation_parts(self, pred_dict, img_metas, targets, save_target_dir, extra_dict):
        save_filename = os.path.join(save_target_dir, str(self.iter))
        pred_mask_parts = pred_dict["pred_mask_parts"][0]
        pred_box_parts = pred_dict["pred_box_parts"][0]
        pred_scores = pred_dict["pred_scores"][0]
        gt_mask = maskUtils.decode(targets["mask"][0])
        pred_mask = pred_dict["pred_mask"][0]
        gt_box = targets["bbox"][0]

        if "img_selected_points" in extra_dict:
            img_selected_points = extra_dict["img_selected_points"][0]
        if "q2t_attn_map" in extra_dict:
            q2t_attn_map = extra_dict["q2t_attn_map"][0]

        expression = img_metas[0]["expression"]
        file_name = img_metas[0]["filename"]

        row_columns = self.visualize_params["row_columns"]
        # 将mask张量的第一个维度分割成 2 行 5 列
        mask = pred_mask_parts.reshape(
            row_columns[0],
            row_columns[1],
            pred_mask_parts.shape[-2],
            pred_mask_parts.shape[-1],
        )
        # 将mask转换为numpy数组以便使用matplotlib绘制
        mask_np = mask
        # 创建一个新图像
        fig, axs = plt.subplots(
            row_columns[0],
            row_columns[1],
            figsize=(row_columns[1] * 3, row_columns[0] * 3),
        )
        # 对每一行进行水平拼接
        for i in range(row_columns[0]):
            for j in range(row_columns[1]):
                idx = i * row_columns[1] + j
                axs[i, j].imshow(mask_np[i, j], cmap="gray")
                axs[i, j].axis("off")
                box = pred_box_parts[idx]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                axs[i, j].add_patch(rect)
                # 添加score值
                score_text = f"{pred_scores[idx]:.2f}"
                axs[i, j].text(
                    box[0],
                    box[1] - 10,
                    score_text,
                    color="red",
                    fontsize=12,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                )
                if "img_selected_points" in extra_dict:
                    select_point = img_selected_points[idx].cpu().detach().numpy() * gt_mask.shape[-1]
                    # 绘制 select_point，使用一个小圆圈表示选中的点
                    circle = patches.Circle(select_point, radius=5, color="red", fill=True)  # 调整颜色和半径
                    axs[i, j].add_patch(circle)
                if "q2t_attn_map" in extra_dict:
                    tokenized_words = img_metas[0]["tokenized_words"]
                    text_attn_map = q2t_attn_map[idx]
                    if len(tokenized_words) < len(text_attn_map) - 2:
                        targeted_text_attn_map = text_attn_map[1 : len(tokenized_words) + 1]
                    else:
                        targeted_text_attn_map = text_attn_map[1:-1]
                    targeted_word = tokenized_words[torch.argmax(targeted_text_attn_map)]
                    axs[i, j].text(
                        0.5,
                        -0.1,
                        targeted_word,  # 显示在图片上方
                        color="blue",
                        fontsize=12,
                        fontweight="bold",
                        verticalalignment="top",
                        horizontalalignment="center",
                        transform=axs[i, j].transAxes,  # 使用 Axes 的坐标系
                    )
        save_filename_query = save_filename + "-{}-querymap.jpg".format(expression)
        plt.savefig(save_filename_query)

        if "attn_map" in extra_dict:
            attn_maps = extra_dict["attn_map"]
            attn_maps = (
                F.interpolate(attn_maps, size=(224, 224), mode="bilinear", align_corners=True).cpu().detach().numpy()
            )
            attn_maps = attn_maps[0].reshape(
                row_columns[0],
                row_columns[1],
                attn_maps.shape[-2],
                attn_maps.shape[-1],
            )
            fig, axs = plt.subplots(
                row_columns[0],
                row_columns[1] if row_columns[1] * row_columns[0] != 1 else row_columns[1] + 1,
                figsize=(row_columns[1] * 3, row_columns[0] * 3),
            )
            for i in range(row_columns[0]):
                for j in range(row_columns[1]):
                    axs[i, j].imshow(attn_maps[i, j], cmap="hot")
                    axs[i, j].axis("off")

        save_filename_attnmap = save_filename + "-{}-attnmap.jpg".format(expression)
        plt.savefig(save_filename_attnmap)

        save_filename_src = save_filename + "-{}-src.jpg".format(expression)

        H, W = gt_mask.shape
        box_gt = (gt_box.cpu().detach().numpy()).astype(np.int32)
        mask_gt = gt_mask.astype(np.int32)
        mask_gt = Image.fromarray(mask_gt * 255)
        image_gt = Image.new("RGB", (W, H))
        image_gt.paste(mask_gt)
        draw_gt = ImageDraw.Draw(image_gt)
        for box in box_gt:
            draw_gt.rectangle(list(box), outline="red", width=2)

        filterd_pred_box = pred_box_parts[np.where(pred_scores > self.box_threshold)]
        box_pred = filterd_pred_box.astype(np.int32)
        mask_pred = pred_mask.astype(np.int32)
        mask_pred = Image.fromarray(mask_pred * 255)
        image_pred = Image.new("RGB", (W, H))
        image_pred.paste(mask_pred)
        draw_pred = ImageDraw.Draw(image_pred)
        for box in box_pred:
            draw_pred.rectangle(list(box), outline="blue", width=2)

        imshow_image_nums = 3

        if "attn_map_query" in extra_dict:
            attn_map = extra_dict["attn_map_query"][0]
            attn_map = attn_map.cpu().detach().numpy()
            attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))
            attn_map = (attn_map * 255).astype(np.uint8)
            attn_map = cv2.resize(attn_map, (W, H))
            attn_map = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
            attn_map = cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB)

            # Draw the points on the attention map
            points_list = img_selected_points.cpu().detach().numpy() * gt_mask.shape[-1]
            attn_map_with_point = deepcopy(attn_map)
            for point in points_list:
                cv2.circle(
                    attn_map_with_point,
                    point.astype(np.int32),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1,
                )
            attn_map_with_point = Image.fromarray(attn_map_with_point)
            attn_map = Image.fromarray(attn_map)
            image_attn = Image.new("RGB", (W, H))
            image_attn.paste(attn_map)
            image_attn_with_point = Image.new("RGB", (W, H))
            image_attn_with_point.paste(attn_map_with_point)

            imshow_image_nums = 5

        img_source = Image.open(file_name)
        img_source = img_source.resize((W, H))
        concat_image = Image.new("RGB", (W * imshow_image_nums + (imshow_image_nums - 1) * 10, H), "white")
        concat_image.paste(img_source, (0, 0))
        concat_image.paste(image_gt, (W + 10, 0))
        concat_image.paste(image_pred, (2 * W + 20, 0))
        if "attn_map_query" in extra_dict:
            concat_image.paste(image_attn, (3 * W + 30, 0))
            concat_image.paste(image_attn_with_point, (4 * W + 40, 0))

        concat_image.save(save_filename_src)
