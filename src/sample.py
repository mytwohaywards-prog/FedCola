import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from sklearn.cluster import KMeans
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# --- 配置参数 ---
COCO_ROOT = './data/coco'  # 你的 COCO 数据根目录
IMG_DIR = os.path.join(COCO_ROOT, 'all_images')  # 使用 val2014 作为候选池
ANN_FILE = os.path.join(COCO_ROOT, 'annotations/captions_val2014.json')
# 输出文件模板，程序会自动生成 _v0.json, _v1.json 等
OUTPUT_FILE_TEMPLATE = './data/coco/fedif_golden_set.json'

TARGET_SIZE = 1000  # 最终需要的样本数量 (即 N 类)
CLIP_THRESHOLD = 0.2  # 图文匹配度阈值 (低于此值的视为脏数据)
BATCH_SIZE = 128  # 推理时的 Batch Size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_VERSIONS = 5  # [新增] 生成 5 份不同的黄金集版本


class COCOCandidateDataset(Dataset):
    """简单的 COCO 数据加载器，用于批量提取特征"""

    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        caption = ann['caption']
        img_id = ann['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        full_path = os.path.join(self.img_dir, path)

        try:
            image = Image.open(full_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # 简单的容错处理，防止单张图片损坏导致崩溃
            print(f"Warning: Failed to load image {full_path}. Error: {e}")
            # 返回全黑图或跳过，这里简单返回 None 由 collate_fn 处理或报错
            # 为简化逻辑，这里假设数据基本完好
            image = torch.zeros(3, 224, 224)

        return image, caption, full_path


def main():
    print(f"Loading CLIP model on {DEVICE}...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    # 1. 加载全量候选数据
    print(f"Loading COCO dataset from {ANN_FILE}...")
    dataset = COCOCandidateDataset(IMG_DIR, ANN_FILE, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 特征提取 & CLIP 过滤 (Filter)
    # 这部分比较耗时，只需要做一次
    print("Step 1: Extracting features and filtering by CLIP score...")
    high_quality_samples = []

    with torch.no_grad():
        for images, captions, paths in tqdm(dataloader):
            images = images.to(DEVICE)
            text_tokens = clip.tokenize(captions, truncate=True).to(DEVICE)

            # 提取特征
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            # 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 计算相似度 (Batch wise)
            similarity = (image_features * text_features).sum(dim=-1)

            # 过滤
            for i in range(len(paths)):
                score = similarity[i].item()
                if score > CLIP_THRESHOLD:
                    high_quality_samples.append({
                        "path": paths[i],
                        "caption": captions[i],
                        "feature": image_features[i].cpu().numpy(),
                        "score": score
                    })

    print(f"  -> Original samples: {len(dataset)}")
    print(f"  -> High quality samples (Score > {CLIP_THRESHOLD}): {len(high_quality_samples)}")

    if len(high_quality_samples) < TARGET_SIZE:
        print(f"Error: Not enough high quality samples! Lower the threshold or use more data.")
        return

    # 3. [核心修改] 循环生成多个版本的 Core-Set
    print(f"\nStep 2: Generating {NUM_VERSIONS} versions of Golden Set using K-Means...")

    # 准备特征矩阵 (N, D) - 只需准备一次
    features_matrix = np.vstack([item['feature'] for item in high_quality_samples])

    for v in range(NUM_VERSIONS):
        print(f"\n--- [Version {v}] Clustering ---")

        # [关键] 每次使用不同的随机种子，导致 K-Means 初始化不同，聚类中心也会发生微小偏移
        current_seed = 42 + v
        kmeans = KMeans(n_clusters=TARGET_SIZE, random_state=current_seed, n_init=10)
        kmeans.fit(features_matrix)

        # 4. 中心采样 (Center Sampling)
        selected_indices = []
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        for cluster_id in range(TARGET_SIZE):
            # 找到属于该聚类的所有样本的索引
            member_indices = np.where(labels == cluster_id)[0]

            if len(member_indices) == 0:
                continue

            # 计算该聚类内所有样本到聚类中心的距离
            member_features = features_matrix[member_indices]
            center = centers[cluster_id]
            distances = np.linalg.norm(member_features - center, axis=1)

            # 选择距离最近的那个样本的原始索引
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

        # 构造带版本号的文件名，例如 fedif_golden_set_v0.json
        output_filename = OUTPUT_FILE_TEMPLATE.replace('.json', f'_v{v}.json')
        print(f"Step 3 (v{v}): Saving {len(final_dataset)} samples to {output_filename}...")

        with open(output_filename, 'w') as f:
            json.dump(final_dataset, f, indent=4)

    print("\nAll Done! You have generated multiple golden sets for rotation.")


if __name__ == "__main__":
    main()