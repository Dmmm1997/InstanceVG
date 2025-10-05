import time
import copy
import numpy
import torch
import random

from instancevg.apis.test import grec_evaluate_f1_nacc, gres_evaluate

from .test import accuracy
from instancevg.datasets import extract_data
from instancevg.utils import get_root_logger, reduce_mean, is_main
from collections import defaultdict
import wandb

try:
    import apex
except:
    pass


def set_random_seed(seed, deterministic=False):
    """Args:
    seed (int): Seed to be used.
    deterministic (bool): Whether to set the deterministic option for
        CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
        to True and `torch.backends.cudnn.benchmark` to False.
        Default: False.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(epoch, cfg, model, optimizer, loader):
    model.train()

    if cfg.distributed:
        loader.sampler.set_epoch(epoch)

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    loss_det_list, loss_mask_list, loss_label_nt_list = [], [], []
    for batch, inputs in enumerate(loader):
        data_time = time.time() - end
        if not cfg.distributed:
            inputs = extract_data(inputs)
        losses, predictions = model(**inputs, epoch=epoch, rescale=False)

        loss_det = losses.pop("loss_det", torch.tensor([0.0], device=device))
        loss_mask = losses.pop("loss_mask", torch.tensor([0.0], device=device))
        loss_label_nt = losses.pop("loss_label_nt", torch.tensor([0.0], device=device))
        loss = loss_det + loss_mask + loss_label_nt
        # loss = loss_det
        optimizer.zero_grad()
        if cfg.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if cfg.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
        optimizer.step()

        if cfg.distributed:
            loss_det = reduce_mean(loss_det)
            loss_mask = reduce_mean(loss_mask)
            loss_label_nt = reduce_mean(loss_label_nt)
        loss_det_list.append(loss_det.item())
        loss_mask_list.append(loss_mask.item())
        loss_label_nt_list.append(loss_label_nt.item())

        if is_main():
            if (batch + 1) % cfg.log_interval == 0 or batch + 1 == batches:
                logger = get_root_logger()
                logger.info(
                    f"train - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                    + f"time: {(time.time()- end):.2f}, data_time: {data_time:.2f}, "
                    + f"loss_det: {sum(loss_det_list) / len(loss_det_list) :.4f}, "
                    + f"loss_mask: {sum(loss_mask_list) / len(loss_mask_list):.4f}, "
                    + f"loss_nt: {sum(loss_label_nt_list) / len(loss_label_nt_list):.4f}, "
                    + f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                )

                wandb.log(
                    {
                        "loss_det": sum(loss_det_list) / len(loss_det_list),
                        "loss_mask": sum(loss_mask_list) / len(loss_mask_list),
                        "loss_label_nt": sum(loss_label_nt_list) / len(loss_label_nt_list),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

        end = time.time()
