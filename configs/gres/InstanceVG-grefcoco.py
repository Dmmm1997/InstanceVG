_base_ = [
    "../_base_/datasets/segmentation/grefs.py",
    "../_base_/misc.py",
]
dataset = "GRefCOCO"
max_token = 50
img_size = 320
patch_size = 16

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFileGRES",
        max_token=max_token,
        with_mask=True,
        with_bbox=True,
        dataset=dataset,
        use_token_type="beit3",
    ),
    # dict(type="LargeScaleJitter", out_max_size=img_size, jitter_min=0.3, jitter_max=1.4),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    # dict(type="Pad", size_divisor=32),
    # dict(type='SampleMaskVertices', num_ray=18, center_sampling=False),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=[
            "img",
            "ref_expr_inds",
            "text_attention_mask",
            "gt_mask_rle",
            "gt_bbox",
            "gt_mask_parts_rle",
        ],
        meta_keys=[
            "filename",
            "expression",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "gt_ori_mask",
            "target",
            "empty",
        ],
    ),
]

val_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFileGRES",
        max_token=max_token,
        with_mask=True,
        with_bbox=True,
        dataset=dataset,
        use_token_type="beit3",
    ),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    # dict(type="Pad", size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=[
            "img",
            "ref_expr_inds",
            "text_attention_mask",
            "gt_mask_rle",
            "gt_bbox",
            "gt_mask_parts_rle",
        ],
        meta_keys=[
            "filename",
            "expression",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "gt_ori_mask",
            "target",
            "empty",
        ],
    ),
]
test_pipeline = val_pipeline.copy()

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=val_pipeline,
    ),
    testA=dict(
        pipeline=test_pipeline,
    ),
    testB=dict(
        pipeline=test_pipeline,
    ),
)

model = dict(
    type="MIXGrefUniModel",
    vis_enc=dict(
        type="BEIT3",
        img_size=img_size,
        patch_size=patch_size,
        vit_type="base",
        drop_path_rate=0.1,
        vocab_size=64010,
        freeze_layer=-1,
        vision_embed_proj_interpolate=False,
        pretrain="pretrain_weights/beit3_base_patch16_224.zip",
    ),
    lan_enc=None,
    fusion=None,
    head=dict(
        type="UniGRefDeformableHead",
        input_channels=768,
        hidden_channels=256,
        query_generation_type="aqsm",
        score_text_select=True,
        dist_guided_query_select_params={"enable": True, "W_dist": 0.003},
        num_queries=10,
        detr_loss={
            "criterion": {"loss_class": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            "matcher": {"cost_class": 1.0, "cost_bbox": 5.0, "cost_giou": 2.0, "cost_point": 2.0},
        },
        loss_weight={"mask": {"dice": 1.0, "bce": 1.0, "nt": 0.2, "neg": 0.2}, "bbox": 0.1},
        # loss_weight={"mask": {"dice": 0.0, "bce": 0.0, "nt":0.0}, "bbox": 0.1},
    ),
    post_params={
        "score_weighted": True,
        "mask_threshold": 0.5,
        "score_threshold": 0.7,
        "with_nms": False,
        "outmask_type": "merge",
    },
    process_visual=True,
    visualize_params={"row_columns": (2, 5)},
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
# work_dir = "work_dir/paper_exp/decoder_ablation/ViTBaseP32-1.0decoder-40ep-512hw-refcocounc"

lr = 0.0005
optimizer_config = dict(
    type="Adam",
    lr=lr,
    lr_vis_enc=lr / 10.0,
    lr_lan_enc=lr,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0,
    amsgrad=True,
)

scheduler_config = dict(
    type="MultiStepLRWarmUp",
    warmup_epochs=1,
    decay_steps=[7],
    decay_ratio=0.1,
    max_epoch=10,
)

log_interval = 50

start_save_checkpoint = 9
