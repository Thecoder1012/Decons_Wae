# ✨ Concurrent Density Estimation with Wasserstein Autoencoders: Generation Fidelity and Adversarial Robustness

## 🗂️ Overview
This repository contains code and resources for training Wasserstein Autoencoders (WAE) to explore generation fidelity and adversarial robustness. The project focuses on concurrent density estimation using different latent space distributions and various activation functions. 

![Datasets](https://github.com/Thecoder1012/Decons_Wae/blob/main/assets/dataset(2).png)

## ⚙️ Getting Started

### 1️⃣ Install Dependencies
Before running any experiments, ensure you have all the required Python packages installed. You can install them with:

```bash
pip install -r requirements.txt
```

### 2️⃣ Training the Model

You can customize the training process with the following arguments:

- **`--groupsort`**: Use Groupsort activation function. Default is `0`. Set to `1` to enable.
- **`--js`**: Use Jensen-Shannon Divergence. Default is `0` (MMD is selected by default). Set to `1` to enable.
- **`--beta`**: Change latent space distribution to Beta. Default is `0` (Gaussian latent space is selected).
- **`--exp`**: Change latent space distribution to Exponential. Default is `0` (Gaussian latent space is selected).
- **`--gauss`**: Opt for Gaussian Ball distribution.
- **`--mnist`**: Use MNIST dataset.

To start training, use the following command:

```bash
python3 train.py
```

### 3️⃣ Swiss Roll Experiments

To run experiments on the Swiss Roll dataset, navigate to the `./swiss_role` directory and execute:

```bash
python3 code_v6.py
```

Ensure that `dataset.py` is in the same folder. Modify it as needed for your experiments.

![Swiss Roll](https://github.com/user-attachments/assets/67582f12-da1e-44f5-a228-6cb9462dbe30)

## 🛡️ Robustness Testing

Robust codes for Gaussian Ball and MNIST datasets are provided in the `Robust` folder. These have been tested with **Cauchy**, **Dirichlet**, and **Gaussian** noise. To run robust experiments on the Swiss Roll dataset, navigate to `./Robust/swiss_role/` and run:

```bash
python3 robust_code.py
```

## 🧪 Experimentation

### ⚙️ Gaussian Ball Clusters
To experiment with the Gaussian Ball position and increase the number of clusters, modify the `dataset.py` file.

### 🔄 MMD and JS Loss Integration
We recommend setting a portion of 0.2 for MMD and JS integration with the reconstruction loss. You can adjust this ratio in `config.py`.

### 🎛️ Hyperparameter Tuning
We encourage open collaboration on hyperparameter settings. All configurations are available in `config.py` for easy modification and experimentation.

### 🧠 Model Architecture
The `model.py` file contains a simple dense neural network model for MNIST and Gaussian Ball reconstruction.

### 🧩 Robustness Ratios
In robustness testing, datasets are mixed with specific ratios:
- **Gaussian Ball**: See lines **127, 129** in `cauchy.py`.
- **Dirichlet**: See lines **134, 137** in `dirichlet.py`.

Adjust the ratios on lines **95** and **101** respectively. Similar modifications can be made for the MNIST dataset.
