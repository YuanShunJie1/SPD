# SPD: Shallow Backdoor Protecting Deep Backdoor Against Backdoor Detection

This repository provides the official implementation of our paper **"SPD: Shallow Backdoor Protecting Deep Backdoor Against Backdoor Detection"**. *This code is released for academic and research purposes only. Misuse of backdoor techniques may lead to serious ethical and legal consequences.*

## 🔥 Overview
Backdoor attacks pose serious threats to the integrity of Deep Neural Networks (DNNs). However, most existing backdoor attacks are vulnerable to **backdoor detection methods** and **human visual inspection**. To address this, we propose a backdoor attack method named **SPD (Shallow Protecting Deep)**.
SPD implants:
- A **deep backdoor** in the **frequency domain**, which is imperceptible and encoded using an autoencoder.
- A **shallow backdoor** in the **pixel domain**, which distracts the defender and protects the deep backdoor.

## 🚀 Usage
### 1️⃣ Step 1: Prepare Data
Before launching the attack, make sure you have downloaded the datasets (CIFAR-10, GTSRB, ImageNet-12). You may modify the dataloader scripts if your dataset path is customized:
```bash
- dataloader_cifar.py
- dataloader_gtsbr.py
- dataloader_imagenet12.py
```

### 2️⃣ Step 2: Inject Backdoor
Run the following commands to execute SPD:
  ```bash
  python spd_cifar10.py #CIFAR-10
  python spd_gtsbr.py #GTSBR
  python spd_imagenet12.py #ImageNet-12
  ```

## 🙏 Acknowledgements

We sincerely thank the authors and contributors of previous works in the field of backdoor attacks and defenses. In particular, we would like to acknowledge BackdoorBench (https://github.com/SCLBD/BackdoorBench). 
