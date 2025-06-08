# Blackbox, Whitebox & Label-Only Membership Inference Attacks

## Overview

This repository contains implementations of three types of **Membership Inference Attacks (MIA)** on machine learning models:

* **Blackbox Membership Inference Attack** ([Shokri et al., 2017](https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf))
* **Whitebox Membership Inference Attack** ([Nasr et al., 2019](https://arxiv.org/abs/1907.09173))
* **Label-Only Membership Inference Attack** ([Li et al., 2021](https://yangzhangalmo.github.io/papers/CCS21-Label.pdf))

These attacks are performed on three datasets:

* **CIFAR-10** (Image Classification)
* **MNIST** (Handwritten Digit Recognition)
* **CelebA** (Facial Attribute Classification)

## Attack Types

### 1. Blackbox Membership Inference Attack

Reference: [Shokri et al., 2017](https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf)

* Uses shadow models to approximate the target model's decision boundary
* Extracts confidence scores from the target model and trains an attacker model on confidence scores

### 2. Whitebox Membership Inference Attack

Reference: [Nasr et al., 2019](https://arxiv.org/abs/1907.09173)

* Requires access to the target model's internal parameters and gradients
* Can leverage gradient ascent techniques to increase attack effectiveness

### 3. Label-Only Membership Inference Attack

Reference: [Li et al., 2021](https://yangzhangalmo.github.io/papers/CCS21-Label.pdf)

* Works without confidence scores, using only predicted labels
* Compares decision boundaries between target and auxiliary models

## How to Use

### 1. Running the Attack

Execute the main script and select the attack type and dataset:

```bash
python main.py
```

You will be prompted to enter the attack type (`blackbox`, `whitebox`, or `label_only`) and the dataset (`CIFAR-10`, `MNIST`, or `CelebA`).

### 2. Structure of `main.py`

The script performs the following steps:

1. **Loads the target model** based on the selected dataset.
2. **Extracts confidence scores** if attack type = black box.
3. **Loads the attacker model** (only for blackbox and label-only attack) and performs membership inference.
4. **Outputs performance metrics** (Accuracy, Precision, Recall) and example predictions.

## Dependencies

To set up the required dependencies, you can use either **pip** or **conda**.

### Install with pip:

```bash
pip install -r requirements.txt
```

### Install with conda:

```bash
conda env create -f environment.yml
conda activate MIA_Env
```

If you want to manually set up the environment, the key dependencies are:

* `torch`
* `torchvision`
* `numpy`
* `scipy`
* `scikit-learn`
* `matplotlib`
* `pandas`
* `tqdm`
* `requests`
* `jinja2`

### Generating `requirements.txt` and `environment.yml`

If you are using **Conda**, you can generate the `requirements.txt` file from your environment using:

```bash
conda activate MIA_Env
pip freeze > requirements.txt
```

For a full Conda environment export, run:

```bash
conda env export --no-builds > environment.yml
```

These files allow others to reproduce the same environment easily.

## Notes

* Ensure that CUDA is available for optimal performance.
* The whitebox attack is not included in the main file.
