# leichte-sprache2image
Generate your very own "Leichte Sprache" images by creating a dataset and fine-tune SDXL.
This repository will lead you through the process of dataset-creation, fine-tuning, image-generation and evaluation.
It consists of 4 directory:
* dataset-preparation: Scrape "Leichte Sprache" images with their descriptions and prepare folder-structure for fine-tuning. All in all 4 different variations of datasets are created.
* image-generation: Apply the LoRAs on SDXL from diffusers and generate images with the scraped image descriptions as prompts
* evaluation: Apply ImageReward and FID to your generated images
* storage: In this directory all the datasets and generated images are stored as well as an example configuration.toml for LoRA Fine-Tuning. Fell free to change the hyperparameters.
## How to generate "Leichte Sprache" images?
### Create your dataset
1. Install necessary packages
```pip install -r requirements.txt```
2. Create the basic dataset by following the notebook at dataset-preparation/create-dataset.ipynb
3. Process the basic dataset and create different variations of it by following the notebook at dataset-preparation/process-dataset.ipynb
4. (optional) you can find some suggests visualization steps in dataset-preparation/visualize-dataset.ipynb
### Fine-Tune SDXL with LoRA
1. Fill in missing absolut paths in the fine-tuning configuration file storage/basic-lora-config.toml - you can vary the hyperparameters (e.g. reduce the batch size if you cuda runs out of memory)
2. Clone the fine-tuning scripts repository for stable diffusion <br>
```cd ..```  <br>
```git clone https://github.com/kohya-ss/sd-scripts.git``` <br>
and follow the instructions in its README.md to install necessary requirements
3. Start fine-tuning by ```python sdxl_train_network.py --config_file=<absolut-path-to>/basic-lora-config.toml```
### Generate Images
1. run generate_images in image-generation/generate-images.py with the base directory from one of the four dataset varies. <br>
It will generate an image for each test image (&lt;dataset-path&gt;/test) and LoRA-Checkpoint (&lt;dataset-path&gt;/loras). The images will be saved in &lt;dataset-path&gt;/output/&lt;checkpoint-name&gt;.

### Evaluate Images
1. calculate fid-score by running calculate_fid in image-evaluation/frechet-inception-distance.py with the base directroy from one of the four dataset varies. <br>
Keep in mind that you will need to have some generated images in &lt;dataset-path&gt;/output/&lt;checkpoint-name&gt; and &lt;dataset-path&gt;/test-images-only/&lt;checkpoint-name&gt;
2. calculate ImageReward-score by running calculate_image_reward in image-evaluation/image-reward.py with the base directroy from one of the four dataset varies. <br>
Keep in mind that you will need to have some generated images in &lt;dataset-path&gt;/output/&lt;checkpoint-name&gt; and &lt;dataset-path&gt;/test-images-only/&lt;checkpoint-name&gt;
3. (optional) check the example visualization of ImageReward and FID in image-evaluation/visualize-evaluation.ipynb

## DISCLAIMER
This repository has been created as part of a bachelor's thesis titled 'Leichte Sprache und generative KI: Bilder als Unterstützung für Texte in Leichter Sprache.' <br>
The dataset names in this repository follow a slightly different naming strategy. Below is a mapping of the dataset names:

| Name in repository                    | Name in thesis |
|---------------------------------------|----------------|
| translated-description_random-split   | RS_D           |
| generated-description_random-split    | RS_IC          |
| translated-description_category-split | CS_D           |
| generated-description_category-split  | CS_IC          |


