import sys
sys.path.append(r"F:\Sem7\Thesis\CODE\src")
from CosineSimilarity import ImageTextSimilarity
import os

dataset_path = r"F:\Sem7\Thesis\CODE\dataset\test\subtask_a_test.tsv"

if not os.path.exists(dataset_path):
    print(f"Dataset file not found: {dataset_path}")
    exit(1)

try:
    similarity_model = ImageTextSimilarity(dataset_path)
    ranked_results = similarity_model.process_test_data()

    for result in ranked_results[:5]:
        print(f"Sentence: {result['sentence']}")
        print(f"Compound: {result['compound']}")
        print(f"Ranked Images: {result['ranked_images']}\n")
except Exception as e:
    print(f"Error running similarity model: {e}")