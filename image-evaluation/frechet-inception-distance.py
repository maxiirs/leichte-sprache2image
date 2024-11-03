import os
import pandas as pd
from pytorch_fid import fid_score

translated_random_split = "../storage/translated-description_random-split"
generated_random_split = "../storage/generated-description_random-split"
translated_category_split = "../storage/translated-description_category-split"
generated_category_split = "../storage/generated-description_category-split"


def calculate_fid(dir):
    os.makedirs(f"../storage/{dir}/evaluation", exist_ok=True)
    csv_file = f"../storage/{dir}/evaluation/frechet-inception-distance.csv"

    output_path = os.path.abspath(os.path.join(dir, "output"))
    test_path = os.path.abspath(os.path.join(dir, "test-images-only"))

    fid_scores = {}

    for source_name in os.listdir(output_path):
        source_folder = os.path.join(output_path, source_name)
        if os.path.isdir(source_folder):
            try:
                source_folder = os.path.abspath(source_folder)
                fid_value = fid_score.calculate_fid_given_paths(
                    paths=[test_path, source_folder],
                    device="cuda",
                    batch_size=50,
                    dims=2048,
                    num_workers=len(os.sched_getaffinity(0))
                )
                fid_scores[source_name] = fid_value
            except Exception as e:
                print(f"Could not calculate FID for {source_name}: {e}")
                fid_scores[source_name] = None

    df = pd.DataFrame(list(fid_scores.items()), columns=["checkpoint-name", "fid-score"])
    df.to_csv(csv_file, index=False)


#for example
for dir in [translated_random_split, translated_category_split, generated_random_split, generated_category_split]:
    calculate_fid(dir)
