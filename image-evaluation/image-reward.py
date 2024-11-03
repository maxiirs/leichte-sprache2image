import os
import pandas as pd
import ImageReward as RM

translated_random_split = "../storage/translated-description_random-split"
generated_random_split = "../storage/generated-description_random-split"
translated_category_split = "../storage/translated-description_category-split"
generated_category_split = "../storage/generated-description_category-split"


def calculate_image_reward(image_directory):
    model = RM.load("ImageReward-v1.0")
    result_data = []
    output_dir = os.path.join(image_directory, "output")
    csv_file = f"../storage/{dir}/evaluation/image-reward.csv"
    os.makedirs(f"../storage/{dir}/evaluation", exist_ok=True)

    for sample in os.listdir(f"{image_directory}/test"):
        image_paths = []
        title = sample
        prompt_file = os.path.join(f"{image_directory}/test/{sample}", f"{title}.txt")
        png_file = os.path.join(f"{image_directory}/test/{sample}", f"{title}.png")

        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as file:
                prompt = file.read().strip()
        else:
            print(f"could not find {prompt_file}")
            continue

        if os.path.exists(png_file):
            image_paths.append((png_file, f"reference"))
        else:
            print(f"could not find {png_file}")

        for lora in os.listdir(output_dir):
            png_file = os.path.join(output_dir, lora, f"{title}.png")
            if os.path.exists(png_file):
                image_paths.append((png_file, lora))
            else:
                print(f"Output-Bilddatei {png_file} nicht gefunden.")

        if len(image_paths) == 0:
            print(f"could not find generated images for {title}")
            continue

        try:
            image_paths_list = [path[0] for path in image_paths]  # Extrahiere nur die Dateipfade
            image_reward_scores = model.score(prompt, image_paths_list)

            result_row = {"title": title}
            for idx, (image_path, suffix) in enumerate(image_paths):
                result_row[f"{suffix}"] = image_reward_scores[idx]

            result_data.append(result_row)
        except Exception as e:
            print(f"Could not calculate image reward for {title}: {e}")
            continue

    df = pd.DataFrame(result_data)
    df.to_csv(csv_file, index=False)


# for example
for dir in [translated_random_split, translated_category_split, generated_random_split, generated_category_split]:
    calculate_image_reward(dir)
