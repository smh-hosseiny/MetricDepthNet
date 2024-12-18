# **ZeroShotDepth**

## **Zero-shot Metric Depth Estimation Using UniDepth, DepthAnything, and DepthPro**

This project implements a **zero-shot metric depth estimation model** using a combination of different components. The model leverages **UniDepth** as the base architecture, **DepthAnything** for training pipeline, and the **DepthPro** loss function for enhanced accuracy in depth predictions. It has been trained on **four datasets** to perform robust zero-shot depth estimation across different environments.


--------------------------------------------


## **Components**

- **Base Model:** [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) - a state-of-the-art depth estimation model that serves as the foundation of this project.
- **Training Code:** Adapted from [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) for scalable and efficient training across large datasets.
- **Loss Function:** Derived from [DepthPro](https://github.com/apple/ml-depth-pro), the gradient loss function is designed to produce sharp details in the depth map.


## **Model Overview**

This project brings together the strengths of three powerful components.
The resulting model is capable of predicting **monocular zero shot metric depth** without requiring any additional information.



--------------------------------------------


## **Usage Instructions**

### **1. Download Model Checkpoint**

You can download the pre-trained model checkpoint required for inference from [here](https://www.dropbox.com/scl/fi/aw2598t53kwone9au6c7r/latest.pth?rlkey=rzig6c7c4gcay1ve4g1v3ypxr&st=7myvphu6&dl=0).

*Make sure to place the downloaded checkpoint file in the `checkpoint/` directory.*



### **2. Clone the Repository and Set Up the Environment**:
1. Clone the Repository:
```bash
git clone https://github.com/yourusername/ZeroShotDepth.git
cd ZeroShotDepth
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
1. Prepare the datasets and configs
2. Run the training scrpts:
3. 
   ```bash
   bash dist_train.sh 
   ```


--------------------------------------------



## **Results**

Below are examples of an input RGB image and its corresponding predicted depth map:

![Input RGB Image](rgb_depth_comparison.png)









   
