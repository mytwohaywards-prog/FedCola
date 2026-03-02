import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FedIfEvaluator:
    def __init__(self, server):
        self.server = server
        self.device = server.server_device
        # 自动获取 Golden Set 用于评估 (假设 server 已加载)
        if hasattr(server, 'mm_val_loaders') and len(server.mm_val_loaders) > 0:
            self.loader = server.mm_val_loaders[0]
        else:
            logger.warning("[Evaluator] No Golden Set loader found! Analytics disabled.")
            self.loader = None

    def run_full_eval(self, round_idx, save_dir):
        """执行全套评估：Zero-shot + t-SNE"""
        if not self.loader: return
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # 1. 提取特征 (全量用于检索，采样用于 t-SNE)
        img_feats, txt_feats = self._extract_features()
        if img_feats is None: return

        # 2. 计算 Zero-shot Retrieval (核心指标)
        metrics = self._compute_retrieval_metrics(img_feats, txt_feats)
        logger.info(f"[Round {round_idx}] Zero-shot Metrics: {metrics}")

        # 如果有 TensorBoard，记录一下
        if self.server.writer:
            self.server.writer.log({"Retrieval/R_Sum": metrics['R_Sum'], "Retrieval/I2T_R1": metrics['I2T_R1']},
                                   round_idx)

        # 3. 绘制 t-SNE (每 20 轮或最后一轮绘制)
        if round_idx % 20 == 0 or round_idx == self.server.args.R:
            self._plot_tsne(img_feats, txt_feats, round_idx, save_dir)

    def _extract_features(self):
        """提取测试集特征"""
        model_name = self.server.mm_dataset_name
        if not model_name: return None, None

        model = self.server.global_models[model_name]
        model.to(self.device)
        model.eval()

        img_list, txt_list = [], []

        with torch.no_grad():
            for batch in self.loader:
                if isinstance(batch, dict):
                    imgs, txts = batch['image'].to(self.device), batch['text'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    imgs, txts = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                # 获取特征 (feat_out=True)
                outs = model([imgs, txts], feat_out=True)
                # 归一化 (Contrastive Learning 的关键)
                img_list.append(F.normalize(outs[0], dim=-1).cpu())
                txt_list.append(F.normalize(outs[1], dim=-1).cpu())

        model.to('cpu')
        return torch.cat(img_list), torch.cat(txt_list)

    def _compute_retrieval_metrics(self, img_feats, txt_feats):
        """计算 R@1, R@5, R@10"""
        # 计算相似度矩阵 (N x N)
        sims = torch.matmul(img_feats, txt_feats.T).numpy()
        n = sims.shape[0]

        # Image-to-Text
        ranks_i2t = np.zeros(n)
        for i in range(n):
            inds = np.argsort(sims[i])[::-1]  # 降序排列
            # 找到 Ground Truth (对角线元素) 的排名
            ranks_i2t[i] = np.where(inds == i)[0][0]

        # Text-to-Image
        ranks_t2i = np.zeros(n)
        for i in range(n):
            inds = np.argsort(sims[:, i])[::-1]
            ranks_t2i[i] = np.where(inds == i)[0][0]

        def get_r(ranks, k):
            return 100.0 * np.sum(ranks < k) / n

        return {
            "I2T_R1": get_r(ranks_i2t, 1),
            "I2T_R5": get_r(ranks_i2t, 5),
            "T2I_R1": get_r(ranks_t2i, 1),
            "T2I_R5": get_r(ranks_t2i, 5),
            "R_Sum": get_r(ranks_i2t, 1) + get_r(ranks_i2t, 5) + get_r(ranks_t2i, 1) + get_r(ranks_t2i, 5)
        }

    def _plot_tsne(self, img_feats, txt_feats, round_idx, save_dir):
        """绘制 t-SNE 对齐图 (带配对连线)"""
        # 随机采样 300 对，避免图太乱
        n_sample = min(300, len(img_feats))
        idx = torch.randperm(len(img_feats))[:n_sample]

        img_sub = img_feats[idx].numpy()
        txt_sub = txt_feats[idx].numpy()

        # t-SNE 降维
        data = np.concatenate([img_sub, txt_sub], axis=0)
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        emb = tsne.fit_transform(data)

        img_emb = emb[:n_sample]
        txt_emb = emb[n_sample:]

        # 绘图
        plt.figure(figsize=(10, 10))
        plt.scatter(img_emb[:, 0], img_emb[:, 1], c='blue', alpha=0.6, label='Image', s=30)
        plt.scatter(txt_emb[:, 0], txt_emb[:, 1], c='red', alpha=0.6, label='Text', s=30, marker='^')

        # [关键] 绘制连线：连接对应的图文对
        # 如果对齐得好，这些线应该很短；如果对齐得差，线会很长且乱
        for i in range(n_sample):
            plt.plot([img_emb[i, 0], txt_emb[i, 0]], [img_emb[i, 1], txt_emb[i, 1]],
                     color='gray', alpha=0.15, linewidth=0.8)

        plt.legend()
        plt.title(f"Semantic Alignment (Round {round_idx})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"tsne_round_{round_idx}.png"), dpi=300)
        plt.close()


