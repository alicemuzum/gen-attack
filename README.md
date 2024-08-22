# **Gen-Attack: Adversarial Attack Benchmarking Across Generative Models and More**

## **Overview**

This project aims a framework for implementing and evaluating adversarial attacks on deep learning models. The focus is on experimenting with different attack methodologies but right now only FGSM is implemented.

## **Installation**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/alicemuzum/gen-attack.git
   cd gen-attack

2. **Setup**

- Using Docker
  ```bash
  docker compose up --build

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

## **License**
This project is licensed under the MIT License. See the LICENSE file for more details.


