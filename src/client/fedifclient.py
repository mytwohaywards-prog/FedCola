# src/client/fedifclient.py
import torch
import logging
from collections import defaultdict

from .fedavgclient import FedavgClient

logger = logging.getLogger(__name__)


class FedifClient(FedavgClient):
    """
    FedIF Client:
    - 本地训练：完全复用 FedavgClient.update()
    - 训练后：在本地数据上估计“共享 attention 层”的 diagonal Fisher（梯度平方近似）
      -> 存到 self.fisher_matrix，供服务器聚合时读取（retain_model=True 时可直接访问 client 对象）。
    """

    def __init__(self, **kwargs):
        super(FedifClient, self).__init__(**kwargs)

        # 可通过 main.py argparse 注入；没有就用默认值
        self.enable_fisher = getattr(self.args, "enable_fisher", False)
        self.fisher_batches = int(getattr(self.args, "fisher_batches", 2))
        self.fisher_clip = float(getattr(self.args, "fisher_clip", 10.0))
        self.fisher_norm_eps = float(getattr(self.args, "fisher_norm_eps", 1e-6))
        self.fisher_trim_ratio = float(getattr(self.args, "fisher_trim_ratio", 0.2))
        self.fisher_loss_temp = float(getattr(self.args, "fisher_loss_temp", 1.0))

        # server 会读取
        self.fisher_matrix = {}

    def download(self, models):
        super().download(models)


    def update(self):
        """
        1) 先按 FedAvg 本地训练（不改你现有 loss/optimizer）
        2) 再估计 Fisher（只算共享 attention）
        """
        result = super().update()

        #  每轮 Fisher 从 0 开始（放这里更合理）
        self.fisher_matrix = {}

        if self.enable_fisher and self.fisher_batches > 0:
            try:
                self.fisher_matrix = self._compute_diag_fisher(max_batches=self.fisher_batches)
            except Exception as e:
                logger.warning(f"[FedIFClient] Fisher compute failed on client {self.id}: {e}")
                self.fisher_matrix = {}

        logger.info(f"[FedIFClient {self.id}] fisher_keys={len(self.fisher_matrix)}")
        return result
    # -------------------------
    # Fisher helpers
    # -------------------------
    @staticmethod
    def _is_shared_attn_param(name: str) -> bool:
        # 只对共享 attention 估计 Fisher（与你 server 端共享 attn 筛选一致）
        return ("attn.qkv" in name) or ("attn.proj" in name)

    def _compute_diag_fisher(self, max_batches: int = 2):
        """
        diagonal Fisher ≈ E[(∂L/∂θ)^2]，用少量 batch 近似即可
        - 只计算共享 attention(qkv/proj)
        - 做均值归一（除以 batch 数）
        - clip 防数值爆炸
        """
        if max_batches <= 0:
            return {}

        model = self.model.module if hasattr(self.model, "module") else self.model
        model.train()
        model.to(self.device)

        fisher_batches = []
        loss_list = []
        used = 0

        for batch in self.train_loader:
            if used >= max_batches:
                break

            # 兼容 DataLoader 返回结构（img/txt: (x,y)，img+txt: (inputs, targets, ... )）
            model.zero_grad(set_to_none=True)

            if self.modality == "img":
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model([inputs, None])[0]
                loss = self.criterion()(outputs.to(targets.device), targets)

            elif self.modality == "txt":
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model([None, inputs])[1]
                loss = self.criterion()(outputs.to(targets.device), targets)

            elif self.modality == "img+txt":
                # FedavgClient.update() 中是：inputs, targets, _, _, _ = batch :contentReference[oaicite:6]{index=6}
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    raise ValueError("Unexpected batch format for img+txt")

                # inputs 通常是 [img_tensor, txt_tensor] 之类
                if isinstance(inputs, (list, tuple)):
                    inputs = [x.to(self.device) if x is not None else None for x in inputs]
                else:
                    inputs = inputs.to(self.device)

                # targets 在你当前对比学习实现里可能不用，但保持一致
                targets = targets.to(self.device) if hasattr(targets, "to") else targets

                outputs = model([inputs, targets], feat_out=True)
                loss = self.criterion()(*outputs)
            else:
                # 未知 modality：不算 Fisher
                break

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.fisher_clip)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                model.zero_grad(set_to_none=True)
                continue

            # accumulate grad^2
            fisher_snapshot = {}
            for n, p in model.named_parameters():
                if (not p.requires_grad) or (p.grad is None):
                    continue
                if not self._is_shared_attn_param(n):
                    continue
                fisher_snapshot[n] = p.grad.detach().float().pow(2).cpu()

            if fisher_snapshot:
                fisher_batches.append(fisher_snapshot)
                loss_list.append(float(loss.detach().cpu().item()))

            used += 1

        if used == 0 or len(fisher_batches) == 0:
            model.cpu()
            return {}

        loss_arr = torch.tensor(loss_list, dtype=torch.float32)
        trim_ratio = min(max(self.fisher_trim_ratio, 0.0), 0.8)

        if len(loss_arr) > 1 and trim_ratio > 0:
            cut = torch.quantile(loss_arr, q=max(1.0 - trim_ratio, 0.5)).item()
            keep = (loss_arr <= cut).nonzero(as_tuple=False).flatten().tolist()
        else:
            keep = list(range(len(fisher_batches)))

        if len(keep) == 0:
            keep = [int(torch.argmin(loss_arr).item())]

        fisher = defaultdict(lambda: 0.0)
        kept_losses = torch.tensor([loss_list[i] for i in keep], dtype=torch.float32)
        weights = torch.softmax(-kept_losses / max(self.fisher_loss_temp, 1e-6), dim=0)

        for wi, batch_idx in zip(weights.tolist(), keep):
            for n, v in fisher_batches[batch_idx].items():
                fisher[n] = fisher[n] + v * float(wi)

        # mean + clip + cpu
        out = {}
        for n, v in fisher.items():
            if self.fisher_clip > 0:
                v = v.clamp(max=self.fisher_clip)
            scale = 1.0 / (1.0 + float(getattr(self, "noise_gamma", 0.0)))
            out[n] = (v * scale).detach().cpu()

        model.cpu()
        return out
