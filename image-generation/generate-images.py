import csv
import os
from diffusers import AutoPipelineForText2Image, StableDiffusionXLImg2ImgPipeline
import torch
from tqdm import tqdm

translated_random_split = "../storage/translated-description_random-split"
generated_random_split = "../storage/generated-description_random-split"
translated_category_split = "../storage/translated-description_category-split"
generated_category_split = "../storage/generated-description_category-split"


def generate_images(dataset_dir):
    lora_dir = f"{dataset_dir}/loras"
    generation_csv = f"{dataset_dir}/generation.csv"
    output_dir = f"{dataset_dir}/output"

    create_csv_with_descriptions(f"{dataset_dir}")

    with open(generation_csv, newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        samples = list(csv_reader)
        spaltennamen = csv_reader.fieldnames

    lora_files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]

    spaltennamen = extend_csv_columns(spaltennamen, lora_files)

    total_elements = len(samples) * len(lora_files)
    progress_bar = tqdm(total=total_elements, desc="images generated", unit="image")

    for sample in samples:
        title = sample["title"]
        englisch_description = sample["english-description"] + " leichte sprache style"

        for lora_file in lora_files:
            pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
                use_safetensors=True
            ).to("cuda")

            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True,
                variant="fp16"
            ).to("cuda")

            lora_path = os.path.join(lora_dir, lora_file)
            lora_name = os.path.splitext(lora_file)[0]
            pipeline_text2image.load_lora_weights(lora_path)
            image = pipeline_text2image(prompt=englisch_description, width=512, height=512).images[0]

            refined_image = refiner(
                prompt=englisch_description,
                num_inference_steps=40,
                denoising_start=0.8,
                image=image,
            ).images[0]

            output_filename = f"{title}.png"
            output_filepath = os.path.join(f"{output_dir}/{lora_name}", output_filename)

            if not os.path.exists(f"{output_dir}/{lora_name}"):
                os.makedirs(f"{output_dir}/{lora_name}")

            refined_image.save(output_filepath)
            sample[lora_name] = os.path.relpath(output_filepath, output_dir)

            # some cleaning
            del image
            del refined_image
            torch.cuda.empty_cache()

            progress_bar.update(1)

    with open(generation_csv, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=spaltennamen, delimiter=";")
        csv_writer.writeheader()
        csv_writer.writerows(samples)

    progress_bar.close()


def create_csv_with_descriptions(dir):
    rows = []

    for root, _, files in os.walk(f"{dir}/test"):
        png_file = None
        english_txt_file = None

        for file in files:
            if file.endswith(".png"):
                png_file = file
            elif file.endswith(".txt") and not file.endswith("-original.txt"):
                english_txt_file = file

        if png_file and english_txt_file:
            base_name = os.path.splitext(png_file)[0]

            png_path = os.path.join(dir, os.path.relpath(os.path.join(root, png_file), dir))

            with open(os.path.join(root, english_txt_file), "r", encoding="utf-8") as f:
                english_description = f.read().strip().replace("\n", " ")

            rows.append([base_name, png_path, english_description])

    with open(f"{dir}/generation.csv", "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["title", "path-to-test-image", "english-description"])
        csv_writer.writerows(rows)


def extend_csv_columns(spaltennamen, lora_files):
    for lora_file in lora_files:
        spalte_name = os.path.splitext(lora_file)[0]
        if spalte_name not in spaltennamen:
            spaltennamen.append(spalte_name)
    return spaltennamen


# for example
for dir in [translated_random_split, translated_category_split, generated_random_split, generated_category_split]:
    generate_images(dir)
