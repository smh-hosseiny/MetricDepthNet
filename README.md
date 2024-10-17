# **ZeroShotDepth**

## **Zero-shot Metric Depth Estimation Using UniDepth, DepthAnythin, and DepthPro**

This project implements a **zero-shot metric depth estimation model** using a combination of different components. The model leverages **UniDepth** as the base architecture, **DepthAnythin** for training code, and the **DepthPro** loss function for enhanced accuracy in depth predictions. It has been trained on **four public datasets** to perform robust zero-shot depth estimation across different environments.

## **Key Features**

- **Zero-shot learning:** The model doesn't require any fine-tuning on new datasets, making it versatile across different tasks and environments.
- **Metric depth estimation:** Unlike relative depth estimation, this model predicts **real-world distances** (metric depth).
- **Trained on diverse public datasets:** By training on multiple public datasets, the model generalizes well to new scenarios and performs robustly in various applications.

## **Components**

- **Base Model:** [UniDepth]([https://unidepth.example.com](https://github.com/lpiccinelli-eth/UniDepth)) - a state-of-the-art depth estimation model that serves as the foundation of this project.
- **Training Code:** Adapted from [DepthAnything]([https://depthanythin.example.com](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)) for scalable and efficient training across large datasets.
- **Loss Function:** Derived from [DepthPro]([https://depthpro.example.com](https://github.com/apple/ml-depth-pro)), the loss function is designed to maximize accuracy in depth predictions.

## **Model Overview**

This project brings together the strengths of three powerful components:

1. **UniDepth** is known for its flexibility and capability in depth estimation tasks.
2. **DepthAnything's** training pipeline ensures the model can handle various datasets and large-scale training efficiently.
3. **DepthPro's** loss function optimizes the model’s ability to make precise metric depth predictions.

The resulting model is capable of predicting **real-world metric depth** without requiring any additional training on new datasets, making it suitable for a wide range of applications.




## **Usage Instructions**

### **Download Model Checkpoint**

You can download the pre-trained model checkpoint required for inference from [here](https://www.dropbox.com/scl/fi/qokq7nsxa0x8b3alrypjd/latest.pth?rlkey=qi8cxfqf3oib1zx57vs2agal6&st=i9q5qsge&dl=0).

*Make sure to place the downloaded checkpoint file in the `checkpoint/` directory.*


To use this model, follow these instructions:

### **Running Inference**

1. Clone the repository:

   ```bash
   export NAME=MetricDepthNet
   
   git clone https://github.com/yourusername/ZeroShotDepth.git
   cd ZeroShotDepth
   conda create -n $NAME python=3.11
   conda activate $NAME
   pip install -r requirements.txt
   ```
2. Run the model:

   ```bash
   python run.py --load-from checkpoint/latest.pth --max-depth 100 --img-path vis_depth/frame_02.jpg 
   ```



### **Training**

1. Prepare the datasets and configs
2. Run the training scrpts:
3. 
   ```bash
   bash dist_train.sh 
   ```


## **Results**

Below are examples of an input RGB image and its corresponding predicted depth map:

![Input RGB Image](rgb_depth_comparison.png)









   
