import sys
sys.path.append(r"F:\Sem7\Thesis\CODE\src")
from idiomDatasetLoader import IdiomDataset
from vectorComparison import rank_images, evaluate_ranking
import torch

dataset_path = r"F:\Sem7\Thesis\CODE\dataset"
dataset = "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = IdiomDataset(dataset_path=dataset_path, dataset=dataset)
print(f"Total samples: {len(dataset)}")

results = []
for i in range(len(dataset)):
    sample = dataset[i]
    if sample is None:
        continue
    ranked_indices = rank_images(sample["sentence"], sample["image_vectors"], device)
    top1_acc, ndcg = evaluate_ranking(sample["actual_order"], ranked_indices, sample["img_to_idx"])
    results.append({
        "idiom": sample["idiom"],
        "sentence": sample["sentence"],
        "actual_order": sample["actual_order"],
        "ranked_order": [sample["idx_to_img"][i] for i in ranked_indices],
        "top1_acc": top1_acc,
        "ndcg": ndcg
    })

for result in results[:5]:
    print(f"Idiom: {result['idiom']}")
    print(f"Sentence: {result['sentence']}")
    print(f"Actual Order: {result['actual_order']}")
    print(f"Ranked Order: {result['ranked_order']}")
    print(f"Top-1 Accuracy: {result['top1_acc']}")
    print(f"NDCG: {result['ndcg']}\n")
