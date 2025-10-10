# InstanceVG: Improving Generalized Visual Grounding with Instance-aware Joint Learning

<p align="center">
  <a href="https://arxiv.org/pdf/2509.13747" target="_blank"><img src="https://img.shields.io/badge/arXiv-2509.13747-b31b1b.svg?logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/Dmmm997/InstanceVG" target="_blank"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow.svg"></a>
  <a href="https://huggingface.co/datasets/Dmmm997/InstanceVG_Data" target="_blank"><img src="https://img.shields.io/badge/Dataset-Available-brightgreen.svg"></a>
</p>

<p align="center">
  <a href="https://dmmm1997.github.io/">Ming Dai</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=9c_ZcDcAAAAJ">Wenxuan Cheng</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=lpFrrs8AAAAJ">Jiang-Jiang Liu</a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=RLhH0jwAAAAJ&hl=zh-CN">Lingfeng Yang</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=Y6KtijIAAAAJ&hl=zh-CN">Zhenhua Feng</a><sup>4</sup>,
  <a href="https://automation.seu.edu.cn/ywk/list.htm">Wankou Yang</a><sup>1*</sup>,
  <a href="https://jingdongwang2017.github.io/">Jingdong Wang</a><sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>Southeast University &nbsp;&nbsp;
  <sup>2</sup>Baidu VIS &nbsp;&nbsp;
  <sup>3</sup>Jiangnan University &nbsp;&nbsp;
  <sup>4</sup>Nanjing University of Science and Technology
</p>

---

## 📢 News

- **[2025.10.11]** Codes, pretrained models, and datasets are now released! 🎉 .

---

## 🧩 Abstract

Generalized visual grounding tasks, including **Generalized Referring Expression Comprehension (GREC)** and **Segmentation (GRES)**, extend the classical paradigm by accommodating **multi-target** and **non-target** scenarios. While GREC focuses on coarse-level bounding box localization, GRES aims for fine-grained pixel-level segmentation.  

Existing approaches typically treat these tasks **independently**, ignoring the potential benefits of **joint learning** and **cross-granularity consistency**. Moreover, most treat GRES as mere semantic segmentation, lacking **instance-aware reasoning** between boxes and masks.

We propose **InstanceVG**, a **multi-task generalized visual grounding framework** that unifies GREC and GRES via **instance-aware joint learning**. InstanceVG introduces *instance queries* with prior reference points to ensure consistent prediction of **points**, **boxes**, and **masks** across granularities.

To our knowledge, InstanceVG is the **first** framework to jointly tackle both GREC and GRES while integrating instance-aware consistency learning.
Extensive experiments on **10 datasets** across **4 tasks** demonstrate that InstanceVG achieves **state-of-the-art performance**, substantially surpassing existing methods across various evaluation metrics.

---

## 🏗️ Framework Overview

<p align="center">
  <img src="./asserts/instancevg.jpg" width="80%">
</p>

---

## ⚙️ Installation

**Environment requirements**

```bash
CUDA == 11.8
torch == 2.0.0
torchvision == 0.15.1
````

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

InstanceVG depends on components from **[detrex](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html)** and **[detectron2](https://github.com/facebookresearch/detectron2)**.

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule init && git submodule update
pip install -e .
```

Finally, install InstanceVG in editable mode:

```bash
pip install -e .
```

---

## 🧮 Data Preparation

Prepare the **MS-COCO** dataset and download the **referring** and **foreground** annotations from the [Model Zoo](#-model-zoo).

Expected directory structure:

```
data/
└── seqtr_type/
    ├── annotations/
    │   ├── mixed-seg/
    │   │   └── instances_nogoogle_withid.json
    │   ├── grefs/instance.json
    │   ├── ref-zom/instance.json
    │   └── rrefcoco/instance.json
    └── images/
        └── mscoco/
            └── train2014/
```

---

## 🧠 Pretrained Weights

InstanceVG uses **[BEiT-3](https://github.com/microsoft/unilm/blob/master/beit3/README.md)** as both the backbone and multi-modal fusion module.

Download pretrained weights and tokenizer from BEiT-3’s official repository.

```bash
mkdir pretrain_weights
```

Place the following files:

```
pretrain_weights/
├── beit3_base_patch16_224.zip
├── beit3_large_patch16_224.zip
└── beit3.spm
```

---

## 🚀 Demo

**Example 1 — GRES task**

```bash
python tools/demo.py \
  --img "asserts/imgs/Figure_1.jpg" \
  --expression "three skateboard guys" \
  --config "configs/gres/InstanceVG-grefcoco.py" \
  --checkpoint /PATH/TO/InstanceVG-grefcoco.pth
```

**Example 2 — RIS task**

```bash
python tools/demo.py \
  --img "asserts/imgs/Figure_2.jpg" \
  --expression "full half fruit" \
  --config "configs/refcoco/InstanceVG-refcoco.py" \
  --checkpoint /PATH/TO/InstanceVG-refcoco.pth
```

For additional options (e.g., thresholds, alternate checkpoints), see `tools/demo.py`.

---

## 🧩 Training

To train InstanceVG from scratch:

```bash
bash tools/dist_train.sh [PATH_TO_CONFIG] [NUM_GPUS]
```

---

## 📊 Evaluation

To reproduce reported results:

```bash
bash tools/dist_test.sh [PATH_TO_CONFIG] [NUM_GPUS] \
  --load-from [PATH_TO_CHECKPOINT_FILE]
```

---

## 🏆 Model Zoo

All pretrained checkpoints are available on [Model](https://huggingface.co/Dmmm997/InstanceVG).

| Task / Train Set    | Config                                    | Checkpoint                 |
| :------------------ | :---------------------------------------- | :------------------------- |
| RefCOCO/+/g (Base)  | `configs/refcoco/InstanceVG-B-refcoco.py` | `InstanceVG-B-refcoco.pth` |
| RefCOCO/+/g (Large) | `configs/refcoco/InstanceVG-L-refcoco.py` | `InstanceVG-L-refcoco.pth` |
| gRefCOCO            | `configs/gres/InstanceVG-grefcoco.py`     | `InstanceVG-grefcoco.pth`  |
| Ref-ZOM             | `configs/refzom/InstanceVG-refzom.py`     | `InstanceVG-refzom.pth`    |
| RRefCOCO            | `configs/rrefcoco/InstanceVG-rrefcoco.py` | `InstanceVG-rrefcoco.pth`  |

Example reproduction:

```bash
bash tools/dist_test.sh configs/refcoco/InstanceVG-B-refcoco.py 1 \
  --load-from work_dir/refcoco/InstanceVG-B-refcoco.pth
```

---

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@ARTICLE{instancevg,
  author={Dai, Ming and Cheng, Wenxuan and Liu, Jiang-Jiang and Yang, Lingfeng and Feng, Zhenhua and Yang, Wankou and Wang, Jingdong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Improving Generalized Visual Grounding with Instance-aware Joint Learning},
  year={2025},
  doi={10.1109/TPAMI.2025.3607387}
}

@article{dai2024simvg,
  title={SimVG: A Simple Framework for Visual Grounding with Decoupled Multi-Modal Fusion},
  author={Dai, Ming and Yang, Lingfeng and Xu, Yihao and Feng, Zhenhua and Yang, Wankou},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={121670--121698},
  year={2024}
}

@inproceedings{dai2025multi,
  title={Multi-Task Visual Grounding with Coarse-to-Fine Consistency Constraints},
  author={Dai, Ming and Li, Jian and Zhuang, Jiedong and Zhang, Xian and Yang, Wankou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2618--2626},
  year={2025}
}
```

---

## ⭐ Acknowledgements

Our implementation builds upon

* [Detrex](https://github.com/IDEA-Research/detrex)
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3)

We thank these excellent open-source projects for their contributions to the community.
