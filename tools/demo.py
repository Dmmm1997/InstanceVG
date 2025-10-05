# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import re
import mmcv
from mmcv.utils import Config
from instancevg.models import build_model
from instancevg.utils import load_checkpoint
from instancevg.datasets import extract_data
from mmcv.parallel import collate
import os
import cv2
import numpy as np
import pycocotools.mask as maskUtils
import matplotlib.colors as mplc
from transformers import XLMRobertaTokenizer
from instancevg.datasets.pipelines.transforms import Normalize, Resize
from instancevg.datasets.pipelines.formatting import DefaultFormatBundle, CollectData
import matplotlib as mpl
from instancevg.utils.visualizer import GenericMask, VisImage
import matplotlib.patches as mpl_patches


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--img",
        default="asserts/imgs/Figure_2.jpg",
        help="Image file",
    )
    parser.add_argument("--expression", default="full half fruit", help="text")
    parser.add_argument(
        "--config",
        default="configs/refcoco/InstanceVG-B-refcoco/InstanceVG-B-refcoco.py",
        help="Config file",
    )
    parser.add_argument(
        "--img_size",
        default=320,
        help="Img Size",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dir/refcoco/InstanceVG-B-refcoco/InstanceVG-B-refcoco.pth",
        help="Checkpoint file",
    )
    parser.add_argument("--output_dir", default="asserts/outdir", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score_thr", type=float, default=0.7, help="bbox score threshold")
    args = parser.parse_args()
    return args


class instancevg:
    def __init__(self, args):
        self.cfg = Config.fromfile(args.config)
        self.cfg.img = args.img
        self.cfg.expression = args.expression
        self.cfg.output_dir = args.output_dir
        self.cfg.device = args.device
        self.cfg.model["post_params"]["score_threshold"] = args.score_thr
        self.threshold = args.score_thr
        self.model = build_model(self.cfg.model)
        self.model.to(args.device)
        load_checkpoint(self.model, load_from=args.checkpoint)
        self.max_token = 50
        self.tokenizer = XLMRobertaTokenizer("pretrain_weights/beit3.spm")
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        file_client_cfg = dict(backend="disk")
        self.file_client = mmcv.FileClient(**file_client_cfg)
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.Trans_Resize = Resize(
            img_scale=(args.img_size, args.img_size),
            keep_ratio=False,
        )
        self.Trans_Normalize = Normalize(**img_norm_cfg)
        self.Trans_DefaultFormatBundle = DefaultFormatBundle()
        self.Trans_CollectData = CollectData(
            keys=[
                "img",
                "ref_expr_inds",
                "text_attention_mask",
            ],
            meta_keys=["filename", "expression", "ori_shape", "img_shape", "pad_shape", "scale_factor", "empty"],
        )

    def clean_string(self, expression):
        return re.sub(r"([.,'!?\"()*#:;])", "", expression.lower()).replace("-", " ").replace("/", " ")

    def inference_detector(self):
        results = {}
        img, text = self.cfg.img, self.cfg.expression
        img_bytes = self.file_client.get(img)
        image = mmcv.imfrombytes(img_bytes, flag="color", backend=None)
        results["filename"] = img
        results["img"] = image
        results["img_shape"] = image.shape  # (h, w, 3), rgb default
        results["ori_shape"] = image.shape
        results["empty"] = None
        expression = self.clean_string(text)
        tokens = self.tokenizer.tokenize(expression)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if len(tokens) > self.max_token - 2:
            tokens = tokens[: self.max_token - 2]
        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (self.max_token - num_tokens)
        ref_expr_inds = tokens + [self.pad_token_id] * (self.max_token - num_tokens)
        results["ref_expr_inds"] = np.array(ref_expr_inds, dtype=int)
        results["text_attention_mask"] = np.array(padding_mask, dtype=int)
        results["expression"] = expression
        results["max_token"] = self.max_token
        results["with_bbox"] = False
        results["with_mask"] = False
        result_resize = self.Trans_Resize(results)
        result_normalize = self.Trans_Normalize(result_resize)
        result_defaultFormatBundle = self.Trans_DefaultFormatBundle(result_normalize)
        result_CollectData = self.Trans_CollectData(result_defaultFormatBundle)
        data = collate([result_CollectData], samples_per_gpu=1)
        inputs = extract_data(data)
        img_metas = inputs["img_metas"]

        predictions = self.model(**inputs, return_loss=False, rescale=True, with_bbox=True, with_mask=True)

        pred_masks = predictions.pop("pred_masks")
        pred_bboxes = predictions.pop("pred_bboxes")

        for j, (img_meta, pred_mask, pred_bbox) in enumerate(zip(img_metas, pred_masks, pred_bboxes)):
            filename, expression = img_meta["filename"], img_meta["expression"]
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            outfile = os.path.join(self.cfg.output_dir, expression.replace(" ", "_") + "_" + os.path.basename(filename))
            img_level_bboxes = pred_bbox["boxes"]
            pred_score = pred_bbox["scores"]
            refer_ind = pred_score > self.threshold
            refer_bbox = img_level_bboxes[refer_ind]
            self.imshow_expr_mask(filename, pred_mask["pred_masks"], refer_bbox, pred_score[refer_ind], outfile)

    def imshow_expr_mask(self, filename, pred_masks, refer_bbox, pred_scores, outfile):
        facecolor = mplc.to_rgb([0.439, 0.188, 0.627]) + (0.65,)
        edgecolor = mplc.to_rgb([0.0, 0.0, 0.0]) + (1,)
        edgecolor_box = "b"
        img = cv2.imread(filename)[:, :, ::-1]
        height, width = img.shape[:2]
        img = np.ascontiguousarray(img).clip(0, 255).astype(np.uint8)
        output_pred = VisImage(img, scale=1.0)
        if pred_masks is not None:
            if "pred_masks" in pred_masks:
                pred_mask = pred_masks["pred_masks"]
                pred_mask = maskUtils.decode(pred_mask)
            else:
                pred_mask = maskUtils.decode(pred_masks)
            assert pred_mask.shape[0] == height and pred_mask.shape[1] == width
            pred_mask = GenericMask(pred_mask, height, width)
            for segment in pred_mask.polygons:
                polygon = mpl.patches.Polygon(
                    segment.reshape(-1, 2),
                    fill=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=2,
                )
                output_pred.ax.add_patch(polygon)

        if refer_bbox is not None and refer_bbox.shape[0] != 0:
            if len(refer_bbox.shape) == 2:
                pred_bboxes = refer_bbox
            else:
                pred_bboxes = refer_bbox.unsqueeze(0)
            for ind, pred_bbox in enumerate(pred_bboxes):
                pred_bbox_int = pred_bbox.long().cpu().detach().numpy()
                rect = mpl_patches.Rectangle(
                    (pred_bbox_int[0], pred_bbox_int[1]),  # (x1, y1)
                    pred_bbox_int[2] - pred_bbox_int[0],  # width
                    pred_bbox_int[3] - pred_bbox_int[1],  # height
                    linewidth=4,
                    edgecolor="r",
                    facecolor="none",
                )
                output_pred.ax.add_patch(rect)
                if pred_scores is not None:
                    pred_score = pred_scores[ind]
                    output_pred.ax.text(
                        pred_bbox_int[0] - 4,
                        pred_bbox_int[1] + 8,
                        "r:{:.2f}".format(pred_score),
                        fontsize=22,
                        color="r",
                        bbox=dict(facecolor="white", alpha=0.75),
                    )

        cv2.imwrite(outfile, output_pred.get_image()[:, :, ::-1])

    def forward(self):
        self.inference_detector()


if __name__ == "__main__":
    args = parse_args()
    Demo = instancevg(args)
    Demo.forward()
