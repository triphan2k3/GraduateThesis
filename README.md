# Black-box sparse adversarial attack on computer vision models using evolutionary computation.

This is parts of my graduate thesis: **Black-box sparse adversarial attack on computer vision models using evolutionary computation.**

## General information

- **Student 1**: Le Chi Cuong
- **Student 2**: Phan Truong Tri
- **Instructor**: Dr. Luong Ngoc Hoang
- **Grade**: 9/10

## Problem
### Adversarial attack
Given a model **D**, an input image **X**, the task is to find a perturbation **e** such that the output of original image **D(X)** is different from the ouput of purturbed image **D(X+e)**

![alt text](<asset/adversarial attack.png>)

### Black-box sparse attack
In black-box situations, we act as if we didn't know about the target models. The only thing we can do is to get outputs of corresponding inputs.

The **sparse** term indicate that the goodness of pertubation is estimated by the **norm l0** which means the number of elements differing from **0** in pertubation **e** should be as small as possible.

## Method
We employed genetic algorithm to find successful pertubations (i.e the ones make targeted models divert their decisions)
![alt text](<asset/genetic algorithm.png>)
