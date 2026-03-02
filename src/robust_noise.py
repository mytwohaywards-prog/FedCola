from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Sequence, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset


# -------------------------
# 1) client-level noisy profile
# -------------------------
def gen_noisy_clients_vector(args, num_users: Optional[int] = None):
    """
    复刻你参考 noise.py 的 gen_noisy_clients_vector 逻辑：
    - level_n_system: noisy clients fraction
    - 每个 noisy client 的 gamma_c_i ~ U[level_n_lowerb, level_n_upperb]
    返回 gamma_c (每个 client 的噪声强度/概率), gamma_s (0/1 是否 noisy)
    """
    if num_users is None:
        num_users = int(getattr(args, "K", getattr(args, "num_users")))
    seed = int(getattr(args, "seed", 1))
    rng = np.random.RandomState(seed)

    level_n_system = float(getattr(args, "level_n_system", 0.0))
    lowerb = float(getattr(args, "level_n_lowerb", 0.0))
    upperb = float(getattr(args, "level_n_upperb", 0.0))

    gamma_s = np.zeros(num_users, dtype=np.float32)
    gamma_s[: int(level_n_system * num_users)] = 1.0
    rng.shuffle(gamma_s)

    gamma_c_initial = rng.rand(num_users).astype(np.float32)
    gamma_c_initial = (upperb - lowerb) * gamma_c_initial + lowerb
    gamma_c = gamma_s * gamma_c_initial
    return gamma_c, gamma_s


# -------------------------
# 2) Dataset wrappers (LN / DN)
# -------------------------
class LabelNoiseWrapper(Dataset):
    """
    对 classification 的 label 做随机翻转：
    - 每个样本以概率 gamma 被替换为随机类标（0..num_classes-1）
    """
    def __init__(self, base: Dataset, num_classes: int, gamma: float, seed: int = 1):
        self.base = base
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)

        rng = np.random.RandomState(seed)
        self.mask = (rng.rand(len(base)) <= self.gamma)
        self.rand_labels = rng.randint(0, self.num_classes, size=len(base), dtype=np.int64)

        # 透传属性（让 server/client 还能读到）
        for k in ["noise_gamma", "is_noisy", "noise_type"]:
            if hasattr(base, k):
                setattr(self, k, getattr(base, k))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        # 兼容 (x,y) 或 (x,y,...) 的返回
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            return item

        x, y = item[0], item[1]
        if self.mask[idx]:
            new_y = int(self.rand_labels[idx])
            if torch.is_tensor(y):
                y = torch.as_tensor(new_y, dtype=y.dtype, device=y.device)
            else:
                y = new_y

        if isinstance(item, tuple):
            return (x, y, *item[2:])
        return [x, y, *item[2:]]


class DataNoiseWrapper(Dataset):
    """
    对输入 x 加噪（DN）：
    - img: x 加高斯噪声并 clamp
    - txt: token dropout（把 token 置为 pad_id）
    - img+txt: 同时对 image 和 text 做各自噪声
    """
    def __init__(
        self,
        base: Dataset,
        modality: str,
        gamma: float,
        mean: float = 0.0,
        std: float = 0.1,
        seed: int = 1,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        txt_drop_prob: float = 0.1,
        txt_pad_id: int = 0,
        keep_special_tokens: bool = True,  # 保护 [CLS]=101, [SEP]=102 (BERT 常用)
    ):
        self.base = base
        self.modality = modality
        self.gamma = float(gamma)

        self.mean = float(mean)
        self.std = float(std)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self.txt_drop_prob = float(txt_drop_prob)
        self.txt_pad_id = int(txt_pad_id)
        self.keep_special_tokens = bool(keep_special_tokens)

        rng = np.random.RandomState(seed)
        self.mask = (rng.rand(len(base)) <= self.gamma)

        for k in ["noise_gamma", "is_noisy", "noise_type"]:
            if hasattr(base, k):
                setattr(self, k, getattr(base, k))

    def __len__(self):
        return len(self.base)

    def _add_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(x)) or (not torch.is_floating_point(x)):
            return x
        noise = torch.randn_like(x) * self.std + self.mean
        x2 = x + noise
        return torch.clamp(x2, self.clip_min, self.clip_max)

    def _drop_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(tokens)) or (tokens.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long)):
            return tokens
        out = tokens.clone()
        prob = torch.rand(out.shape, device=out.device)
        drop_mask = prob < self.txt_drop_prob

        if self.keep_special_tokens:
            # 常见 special token：101=[CLS], 102=[SEP]
            drop_mask = drop_mask & (out != 101) & (out != 102)

        out[drop_mask] = self.txt_pad_id
        return out

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if not self.mask[idx]:
            return item

        if not isinstance(item, (tuple, list)):
            return item

        # 你的数据在 client.update 里：
        # img/txt: (inputs, targets)
        # img+txt: (inputs, targets, _, _, _) 其中 inputs=image, targets=text tokens
        out = list(item)

        if self.modality == "img":
            out[0] = self._add_gaussian(out[0])
        elif self.modality == "txt":
            out[0] = self._drop_tokens(out[0])
        elif self.modality == "img+txt":
            if len(out) >= 2:
                out[0] = self._add_gaussian(out[0])   # image
                out[1] = self._drop_tokens(out[1])    # text tokens
        return tuple(out)


# -------------------------
# 3) GN: add noise to uploaded state_dict
# -------------------------
def add_gradient_noise_state_dict(
    args,
    state_dict: dict,
    gamma: float,
    seed: int,
):
    """
    参考你 noise.py 里的 add_gradient_noise（对 flattened weights 加高斯噪声），:contentReference[oaicite:9]{index=9}
    这里做成更稳健的版本：
    - 只对 float tensor 加噪（跳过 int/bool buffer）
    - gamma<=0 直接返回
    - 噪声 std 优先用 args.grad_n_std（没有就退回 level_n_std）
    """
    if gamma <= 0:
        return state_dict

    mean = float(getattr(args, "grad_n_mean", getattr(args, "level_n_mean", 0.0)))
    std = float(getattr(args, "grad_n_std", getattr(args, "level_n_std", 0.1)))

    rng = np.random.RandomState(int(seed))
    out = OrderedDict()

    for k, v in state_dict.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            noise = rng.normal(mean, std, size=v.numel()).astype(np.float32)
            noise_t = torch.tensor(noise, dtype=v.dtype).view_as(v)
            out[k] = v + noise_t
        else:
            out[k] = v
    return out
class PairNoiseWrapper(Dataset):
    """
    给 img+txt 检索/对比学习用的“配对噪声”：
    - 以概率 gamma，把样本的 text（或 image）替换成另一个随机样本的，从而制造“假正样本”。
    兼容你 client 端使用的 batch 结构：(img, txt, _, _, _)
    """
    def __init__(self, base: Dataset, gamma: float, seed: int = 1, mode: str = "swap_txt"):
        self.base = base
        self.gamma = float(gamma)
        self.mode = mode

        rng = np.random.RandomState(seed)
        n = len(base)
        self.mask = (rng.rand(n) <= self.gamma)
        self.perm = rng.permutation(n)

        # 尽量避免 perm[i]==i（否则“换成自己”没噪声）
        idxs = np.where(self.mask)[0]
        for i in idxs:
            if self.perm[i] == i:
                j = (i + 1) % n
                self.perm[i], self.perm[j] = self.perm[j], self.perm[i]

        # 透传属性
        for k in ["noise_gamma", "is_noisy", "noise_type"]:
            if hasattr(base, k):
                setattr(self, k, getattr(base, k))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if (not self.mask[idx]) or (not isinstance(item, (tuple, list))) or (len(item) < 2):
            return item

        j = int(self.perm[idx])
        other = self.base[j]
        if (not isinstance(other, (tuple, list))) or (len(other) < 2):
            return item

        x, t = item[0], item[1]
        ox, ot = other[0], other[1]

        if self.mode == "swap_txt":
            new0, new1 = x, ot
        elif self.mode == "swap_img":
            new0, new1 = ox, t
        else:  # swap_both
            new0, new1 = ox, ot

        if isinstance(item, tuple):
            return (new0, new1, *item[2:])
        return [new0, new1, *item[2:]]


# -------------------------
# 4) ADV: FGSM for your MoME-style forward
# -------------------------
def _forward_img_logits(model, images: torch.Tensor):
    """
    适配你 client.update 的 forward 方式：
    img: self.model([inputs, None])[0] :contentReference[oaicite:10]{index=10}
    """
    out = model([images, None])
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.enable_grad()
def fgsm_attack_img(model, images, labels, eps: float, clip_min: float, clip_max: float):
    model.eval()
    images = images.detach()
    images.requires_grad = True

    logits = _forward_img_logits(model, images)
    loss = nn.CrossEntropyLoss()(logits, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    adv = images + eps * images.grad.sign()
    adv = torch.clamp(adv, clip_min, clip_max)
    return adv.detach()


def gen_adv_dataset_img(
    args,
    model,
    dataset: Dataset,
    gamma: float,
    device: str,
    batch_size: int = 64,
):
    """
    参考你 noise.py 的 gen_adv_dataset：先挑 noisy_idx，再 FGSM 生成对抗样本并拼回 TensorDataset :contentReference[oaicite:11]{index=11}
    注意：这会把一个 client 的训练集“物化”成 tensor（适合 CIFAR100 这种小图；不建议直接用在 COCO 这种大集）。
    """
    if gamma <= 0:
        return dataset

    eps = float(getattr(args, "adv_eps", 0.03))
    clip_min = float(getattr(args, "adv_clip_min", -1.0))
    clip_max = float(getattr(args, "adv_clip_max", 1.0))

    rng = np.random.RandomState(int(getattr(args, "seed", 1)) + 9997)
    idxs = np.arange(len(dataset))
    noisy_mask = (rng.rand(len(idxs)) <= gamma)
    noisy_idx = idxs[noisy_mask]
    clean_idx = idxs[~noisy_mask]

    # noisy subset -> generate adv
    images_list, labels_list = [], []

    def _consume_subset(sub_idx, attack: bool):
        if len(sub_idx) == 0:
            return
        sub = Subset(dataset, sub_idx.tolist())
        loader = DataLoader(sub, batch_size=batch_size, shuffle=False)

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if attack:
                x_adv = fgsm_attack_img(model, xb, yb, eps=eps, clip_min=clip_min, clip_max=clip_max)
                images_list.append(x_adv.detach().cpu())
            else:
                images_list.append(xb.detach().cpu())
            labels_list.append(yb.detach().cpu())

    # 先生成 adv，再拼 clean
    _consume_subset(noisy_idx, attack=True)
    _consume_subset(clean_idx, attack=False)

    X = torch.cat(images_list, dim=0)
    Y = torch.cat(labels_list, dim=0)
    return TensorDataset(X, Y)