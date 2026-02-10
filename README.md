# Cross-Domain Image-to-Image Translation with VQGAN and BBDM

This project applies the architecture of the **Brownian Bridge Diffusion Model (BBDM)** to perform cross-domain image-to-image translation on the **CelebAMask-HQ Dataset**. 

The diffusion process is performed entirely in the latent space of a **VQGAN**. This work extends the research presented in the original [BBDM paper](https://arxiv.org/abs/2205.07680) by:
- Adding versatile command-line arguments.
- Training specifically on the CelebAMask-HQ Dataset.
- Incorporating a visualization demo (UI) for real-time testing.

## Usage

All scripts are run through `main.py` with specific flags for different modes. Ensure you have your configuration file (`configs/CelebAMaskHQ-f16.yaml`) and model checkpoint (`CelebAMaskHQ-f16.pth`) ready.

### 1. Training
To train the model from a checkpoint or start fresh:

```bash
python main.py \
  --config configs/CelebAMaskHQ-f16.yaml \
  --train \
  --sample_at_start \
  --save_top \
  --gpu_ids 0 \
  --resume_model CelebAMaskHQ-f16.pth
```

### 2. Sampling on Test Dataset
To run generation on the entire test split of the dataset:

```bash
python main.py \
  --config configs/CelebAMaskHQ-f16.yaml \
  --sample_to_eval \
  --gpu_ids 0 \
  --resume_model CelebAMaskHQ-f16.pth
```

### 3. Inference on a Single Image
To translate a specific image file:

```bash
python main.py \
  --config configs/CelebAMaskHQ-f16.yaml \
  --test \
  --gpu_ids 0 \
  --resume_model CelebAMaskHQ-f16.pth \
  -i "path/to/input/image.jpg" \
  -o "output/path/"
```

### 4. User Interface (Demo)
To launch the interactive visualization tool:

```bash
python main.py \
  --config configs/CelebAMaskHQ-f16.yaml \
  --ui \
  --gpu_ids 0 \
  --resume_model CelebAMaskHQ-f16.pth
```

## 📚 References & Acknowledgements

* **Original Paper:** *Cross-Domain Image-to-Image Translation with VQGAN and BBDM*
* **Acknowledgements:** We extend our gratitude to the authors of the original paper for their foundational work and open-source contributions.
