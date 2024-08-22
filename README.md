# **Gen-Attack: Adversarial Attack Benchmarking Across Generative Models and More**

## **Overview**

This project provides a comprehensive framework for implementing and evaluating adversarial attacks on deep learning models, as well as generating images using state-of-the-art techniques. The focus is on experimenting with different attack methodologies, benchmarking model performance, and exploring image generation through advanced neural networks.

## **Features**

- **Adversarial Attacks:** Implementation of various adversarial attack methods, including Fast Gradient Sign Method (FGSM), with a focus on evaluating their impact on model robustness.
- **Image Generation:** Tools and scripts for generating images using methods such as Stable Diffusion and Generative Adversarial Networks (GANs).
- **Benchmarking:** Includes benchmarking tools to assess model performance against adversarial attacks, with visualizations and results analysis.

## **Directory Structure**

- **`gen_attack/src/adversarial_attacks/`**
  Contains the implementation of adversarial attack methods, including FGSM.

- **`gen_attack/src/benchmarking/`**  
  Includes scripts and notebooks for benchmarking model performance and visualizing results.

- **`gen_attack/src/image-generation/`**  
  Houses scripts and notebooks for generating images using different neural network architectures.

- **`gen_attack/compose.yaml` & `Dockerfile`**  
  Configuration files for setting up the environment using Docker, supporting both CPU and GPU.

## **Installation**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/alicemuzum/gen-attack.git
   cd gen-attack

2. **Setup Environment**

- Using Docker
  ```bash
  docker-compose up --build

- Or setup manually
    ```bash
    poetry install

## **Usage**

### **Adversarial Attacks**
Use the provided scripts in the adversarial_attacks directory to execute and analyze adversarial attacks on various models. Modify the parameters in the scripts to adapt the attack strength and target models.

### **Image Generation**
Explore different image generation techniques using the scripts in the image-generation directory. The provided Jupyter notebooks offer step-by-step guidance.

### **Benchmarking**
Utilize the benchmarking scripts to compare model performance before and after adversarial attacks. The results can be visualized using the visualize.py script.

## **Dependencies**
Python 3.8+
PyTorch
Numpy
Pandas
OpenCV
Matplotlib
Docker (for containerized environments)
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## **License**
This project is licensed under the MIT License. See the LICENSE file for more details.


