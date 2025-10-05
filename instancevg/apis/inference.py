import mmcv
import torch
import os.path as osp

from instancevg.core.utils import imshow_box_mask, imshow_box_mask_parts, is_badcase_boxsegioulowerthanthr
from instancevg.utils import load_checkpoint, get_root_logger
from instancevg.core import imshow_expr_bbox, imshow_expr_mask
from instancevg.models import build_model, ExponentialMovingAverage
from instancevg.datasets import extract_data, build_dataset, build_dataloader

# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import copy

try:
    import apex
except:
    pass


def inference_model(cfg):
    datasets_cfg = []
    for which_set in cfg.which_set:
        datasets_cfg.append(eval(f"cfg.data.{which_set}"))

    datasets = list(map(build_dataset, datasets_cfg))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets))

    model = build_model(cfg.model, word_emb=datasets[0].word_emb, num_token=datasets[0].num_token)
    model = model.cuda()
    if cfg.use_fp16:
        model = apex.amp.initialize(model, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    load_checkpoint(model, load_from=cfg.checkpoint)

    model.eval()
    logger = get_root_logger()
    with_bbox, with_mask = False, False
    for i, which_set in enumerate(cfg.which_set):
        logger.info(f"inferencing on split {which_set}")
        prog_bar = mmcv.ProgressBar(len(datasets[i]))
        with torch.no_grad():
            for batch, inputs in enumerate(dataloaders[i]):
                gt_bbox, gt_mask = [], []
                if "gt_bbox" in inputs:
                    if isinstance(inputs["gt_bbox"], torch.Tensor):
                        inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
                        gt_bbox = copy.deepcopy(inputs["gt_bbox"])
                    else:
                        gt_bbox = copy.deepcopy(inputs["gt_bbox"].data[0])
                    with_bbox = True

                if "gt_mask_rle" in inputs:
                    gt_mask = inputs.pop("gt_mask_rle").data[0]
                    with_mask = True

                if not cfg.distributed:
                    inputs = extract_data(inputs)

                predictions = model(
                    **inputs,
                    return_loss=False,
                    gt_mask=gt_mask,
                    rescale=True,
                    with_bbox=with_bbox,
                    with_mask=with_mask,
                )
                img_metas = inputs["img_metas"]
                batch_size = len(img_metas)

                pred_bboxes = [None for _ in range(batch_size)]
                if with_bbox:
                    pred_bboxes = predictions.pop("pred_bboxes")
                pred_masks = [None for _ in range(batch_size)]
                if with_mask:
                    pred_masks = predictions.pop("pred_masks")
                    if cfg.instance_seg:
                        pred_parts = predictions.pop("parts_list")
                # if cfg["dataset"] == "GRefCOCO":
                tmp_pred_bboxes = []
                for pred_bbox in pred_bboxes:
                    img_level_bboxes = pred_bbox["boxes"]
                    scores = pred_bbox["scores"]
                    keep_ind = scores > cfg.score_threshold
                    img_level_bboxes = img_level_bboxes[keep_ind]
                    tmp_pred_bboxes.append(img_level_bboxes)
                pred_bboxes = tmp_pred_bboxes

                if cfg.instance_seg:
                    tmp_pred_masks_parts = []
                    for pred_mask_part, pred_score in zip(pred_parts["pred_mask_parts"], pred_parts["pred_scores"]):
                        keep_ind = pred_score > cfg.score_threshold
                        img_level_masks = pred_mask_part[keep_ind]
                        tmp_pred_masks_parts.append(img_level_masks)
                    pred_mask_parts = tmp_pred_masks_parts

                for j, (img_meta, pred_bbox, pred_mask, pred_mask_part) in enumerate(
                    zip(img_metas, pred_bboxes, pred_masks, pred_mask_parts)
                ):
                    filename, expression = img_meta["filename"], img_meta["expression"]
                    bbox_gt, mask_gt = None, None
                    if cfg.with_gt and with_bbox:
                        bbox_gt = gt_bbox[j]
                    if cfg.with_gt and with_mask:
                        mask_gt = img_meta["gt_ori_mask"]

                    scale_factors = img_meta["scale_factor"]
                    # pred_bbox /= pred_bbox.new_tensor(scale_factors)

                    outfile = osp.join(
                        cfg.output_dir,
                        cfg.dataset + "_" + which_set,
                        expression.replace(" ", "_") + "_" + osp.basename(filename).split(".jpg")[0],
                    )
                    badcase = False
                    if bbox_gt is not None and len(bbox_gt) >= 2:
                        badcase = True

                    if not cfg.onlybadcase or (cfg.onlybadcase and badcase):
                        # box seg分开绘制
                        # if with_bbox:
                        #     bbox_gt /= bbox_gt.new_tensor(scale_factors)
                        #     outfile_det = outfile + "_box.jpg"
                        #     imshow_expr_bbox(filename, pred_bbox, outfile_det, gt_bbox=bbox_gt)
                        # if with_mask:
                        #     outfile_seg = outfile + "_seg.jpg"
                        #     imshow_expr_mask(filename, pred_mask, outfile_seg, gt_mask=mask_gt, overlay=cfg.overlay)
                        empty = pred_mask["pred_nt"].argmax(dim=0).bool()
                        # boxseg合并绘制
                        outfile_pred_parts = outfile + "_pred_instance.jpg"
                        imshow_box_mask_parts(
                            filename, pred_bbox, pred_mask_part, outfile_pred_parts, empty=empty, gt=False
                        )

                        outfile_pred = outfile + "_pred.jpg"
                        imshow_box_mask(filename, pred_bbox, pred_mask, outfile_pred, gt=False)

                        bbox_gt = bbox_gt.reshape(-1, 4)
                        bbox_gt /= bbox_gt.new_tensor(scale_factors)
                        outfile_gt = outfile + "_gt.jpg"
                        imshow_box_mask(filename, bbox_gt, mask_gt, outfile_gt, gt=True)

                    prog_bar.update()
