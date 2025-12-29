# **ZeroShotDepth**


![UniDepth](https://img.shields.io/badge/Base-UniDepth-blue)
![DepthAnything](https://img.shields.io/badge/Training-DepthAnything-green)
![DepthPro](https://img.shields.io/badge/Loss-DepthPro-purple)
![Zero--Shot](https://img.shields.io/badge/Setting-Zero--Shot-orange)
![Metric Depth](https://img.shields.io/badge/Output-Metric_Depth-red)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-lightgrey)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

<!-- Teaser Prediction Video -->
<p align="center">
  <video src="assets/demo2.mp4"
         controls
         muted
         autoplay
         loop
         width="75%">
    Your browser does not support the video tag.
  </video>
</p>


## **Zero-shot Metric Depth Estimation Using UniDepth, DepthAnything, and DepthPro**

This project implements a **zero-shot metric depth estimation model** using a combination of different components. The model leverages **UniDepth** as the base architecture, **DepthAnything** for training pipeline, and the **DepthPro** loss function for enhanced accuracy in depth predictions. It has been trained on **four datasets** to perform robust zero-shot depth estimation across different environments.


--------------------------------------------

## **Components**

- **Base Model:** [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) — a state-of-the-art architecture for monocular depth estimation.
- **Training Pipeline:** Adapted from [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) for scalable multi-dataset training.
- **Loss Function:** Based on [DepthPro](https://github.com/apple/ml-depth-pro), using gradient-based supervision to preserve sharp geometric details.


## **Model Overview**

By integrating these three components, the model predicts **zero-shot metric depth from a single RGB image**, without requiring camera intrinsics or scene-specific calibration.

--------------------------------------------


## **Usage Instructions**

### **1. Download Model Checkpoint**

You can download the pre-trained model checkpoint required for inference from [here](https://www.dropbox.com/scl/fi/aw2598t53kwone9au6c7r/latest.pth?rlkey=rzig6c7c4gcay1ve4g1v3ypxr&st=7myvphu6&dl=0).

*Make sure to place the downloaded checkpoint file in the `checkpoint/` directory.*



### **2. Clone the Repository and Set Up the Environment**:
1. Clone the Repository:
```bash
git clone https://github.com/smh-hosseiny/MetricDepthNet.git
cd MetricDepthNet
```
2. Create and Activate the Environment:
```bash
conda create -n MetricDepthNet python=3.11
conda activate MetricDepthNet
```
3. Install Required Dependencies:

```bash
pip install -r requirements.txt 
```

### **3. Running Inference**:
Option 1: Command-Line Inference

Infer depth of a specific image:
```bash
python run.py --load-from checkpoint/latest.pth --max-depth 100 --img-path vis_depth/frame_02.jpg
```

Option 2: Using the Gradio Interface

To use the Gradio interface:
 ```bash
python app.py
```
Open the provided local URL in your web browser to interact with the model.


--------------------------------------------
## **Training**
1. Prepare the datasets and configuration files.
2. Run distributed training:
```bash
bash dist_train.sh
```
--------------------------------------------


## **Results**

Examples of an input RGB image and its corresponding predicted metric depth map:

![RGB–Depth Comparison](assets/rgb_depth_comparison.png)







   
