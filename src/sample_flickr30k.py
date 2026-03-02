import os
import json
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# --- 配置参数 (请根据你的实际路径修改) ---
FLICKR_ROOT = './data/flickr30k'
IMG_DIR = os.path.join(FLICKR_ROOT, 'flickr30k-images')
# Flickr30k 的标注通常是这个 .token 文件，或者是 karpathy split 的 json
# 这里假设使用原始 token 文件格式 (results.csv)
ANN_FILE = os.path.join(FLICKR_ROOT, 'results.csv')

# [修改点 1] 输出文件路径模板。程序会自动生成 ..._v0.json, ..._v1.json
OUTPUT_FILE_TEMPLATE = './data/flickr30k/fedif_golden_set.json'

TARGET_SIZE = 1000  # 最终需要的样本数量 (即 N 类)
CLIP_THRESHOLD = 0.2  # 图文匹配度阈值
BATCH_SIZE = 64  # 推理时的 Batch Size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_VERSIONS = 5  # [修改点 2] 生成 5 份不同的黄金集版本


class FlickrCandidateDataset(Dataset):
    """
    Flickr30k 数据加载器 (适配 CSV 格式)
    """

    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        print(f"Parsing annotation file: {ann_file}...")

        # 1. 尝试读取 CSV
        # Flickr30k 的 results.csv 常见格式是使用 '|' 分隔
        try:
            df = pd.read_csv(ann_file)
            if len(df.columns) == 1 and isinstance(df.iloc[0, 0], str) and '|' in df.iloc[0, 0]:
                df = pd.read_csv(ann_file, sep='|', engine='python')
        except Exception as e:
            print(f"CSV read failed, trying default separator '|': {e}")
            df = pd.read_csv(ann_file, sep='|', engine='python')

        # 2. 清理列名
        df.columns = [c.strip() for c in df.columns]

        # 3. 标准化列名映射
        col_map = {
            'image_name': 'image_id',
            'image': 'image_id',
            'comment': 'caption',
            ' caption': 'caption',
            ' comment': 'caption'
        }
        df = df.rename(columns=col_map)

        if 'image_id' not in df.columns or 'caption' not in df.columns:
            print("Warning: Standard column names not found. Using 1st and last columns.")
            df['image_id'] = df.iloc[:, 0]
            df['caption'] = df.iloc[:, -1]

        # 4. 数据清理
        df['caption'] = df['caption'].astype(str)
        df['image_id'] = df['image_id'].astype(str).str.strip()

        # 5. 聚合数据 (Group by Image)
        print("Grouping captions by image...")
        grouped = df.groupby('image_id')['caption'].apply(list).reset_index()

        self.image_names = grouped['image_id'].tolist()
        self.captions_list = grouped['caption'].tolist()

        print(f"Found {len(self.image_names)} unique images.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        captions = self.captions_list[idx]

        full_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(full_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            return None, [], ""

        return image, captions, full_path


def collate_fn_flickr(batch):
    # 过滤掉加载失败的 None
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return torch.Tensor([]), [], []

    images, captions_lists, paths = zip(*batch)
    images = torch.stack(images, 0)
    return images, captions_lists, paths


def main():
    print(f"Loading CLIP model on {DEVICE}...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    # 1. 加载数据
    dataset = FlickrCandidateDataset(IMG_DIR, ANN_FILE, transform=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_flickr
    )

    # 2. 特征提取 & CLIP 过滤 (只做一次，因为这是最耗时的步骤)
    print("Step 1: Extracting features and filtering by CLIP score...")
    high_quality_samples = []

    with torch.no_grad():
        for images, captions_batch_list, paths in tqdm(dataloader):
            if len(images) == 0: continue

            images = images.to(DEVICE)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 对 batch 中的每一张图片进行处理
            for i in range(len(paths)):
                current_captions = captions_batch_list[i]
                if len(current_captions) == 0: continue

                text_tokens = clip.tokenize(current_captions, truncate=True).to(DEVICE)
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # 计算该图片与其 5 个 caption 的相似度
                scores = (image_features[i].unsqueeze(0) * text_features).sum(dim=-1)

                # 找到最高分的 caption
                best_score, best_idx = scores.max(dim=0)
                best_score = best_score.item()
                best_idx = best_idx.item()

                if best_score > CLIP_THRESHOLD:
                    high_quality_samples.append({
                        "path": paths[i],
                        "caption": current_captions[best_idx],
                        "feature": image_features[i].cpu().numpy(),
                        "score": best_score
                    })

    print(f"  -> Total unique images processed: {len(dataset)}")
    print(f"  -> High quality samples (Score > {CLIP_THRESHOLD}): {len(high_quality_samples)}")

    if len(high_quality_samples) < TARGET_SIZE:
        print(f"Error: Not enough high quality samples! Found {len(high_quality_samples)}, needed {TARGET_SIZE}.")
        return

    # 3. [核心修改] 循环生成多个版本的 Core-Set
    print(f"\nStep 2: Generating {NUM_VERSIONS} versions of Golden Set using K-Means...")

    # 准备特征矩阵 (N, D) - 避免在循环中重复构建
    features_matrix = np.vstack([item['feature'] for item in high_quality_samples])

    for v in range(NUM_VERSIONS):
        print(f"\n--- [Version {v}] Clustering ---")

        # [关键] 每次使用不同的随机种子 (42 + v)
        # 这会导致 K-Means 的初始化中心不同，最终选出的 1000 个代表性样本也会有微妙差异
        current_seed = 42 + v
        kmeans = KMeans(n_clusters=TARGET_SIZE, random_state=current_seed, n_init=10)
        kmeans.fit(features_matrix)

        # 4. 中心采样 (Center Sampling)
        selected_indices = []
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        for cluster_id in range(TARGET_SIZE):
            member_indices = np.where(labels == cluster_id)[0]
            if len(member_indices) == 0: continue

            member_features = features_matrix[member_indices]
            center = centers[cluster_id]
            distances = np.linalg.norm(member_features - center, axis=1)

            best_member_idx = np.argmin(distances)
            original_idx = member_indices[best_member_idx]
            selected_indices.append(original_idx)

        # 5. 保存当前版本的结果
        final_dataset = []
        for idx in selected_indices:
            item = high_quality_samples[idx]
            final_dataset.append({
                "image_path": item['path'],
                "caption": item['caption'],
                "clip_score": item['score']
            })

        # 构造带版本号的文件名，例如 ./data/flickr30k/fedif_golden_set_v0.json
        output_filename = OUTPUT_FILE_TEMPLATE.replace('.json', f'_v{v}.json')

        # 确保目录存在
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        print(f"Step 3 (v{v}): Saving {len(final_dataset)} samples to {output_filename}...")
        with open(output_filename, 'w') as f:
            json.dump(final_dataset, f, indent=4)

    print(f"\nAll Done! Generated {NUM_VERSIONS} golden sets for random rotation.")


if __name__ == "__main__":
    main()