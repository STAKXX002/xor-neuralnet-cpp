# XOR Neural Network in C++

This project is a **from-scratch implementation** of a simple feedforward neural network in C++ that learns the classic **XOR problem**.  

It demonstrates the fundamentals of:
- Feedforward neural networks  
- Backpropagation  
- Activation functions  
- Error calculation and gradient descent  

---

## Features
- Written in **pure C++** (no external ML libraries)  
- Uses `tanh` as the activation function  
- Implements **mean squared error (MSE)** as the loss function  
- Includes detailed comments explaining the **math behind forward & backward propagation**  
- Shows how a neural network can learn a **non-linear function (XOR)**  

---

## Example Output
After training, the network correctly predicts XOR logic:

```
Inputs: 0 0 → Output: ~0.02 (Target: 0)  
Inputs: 0 1 → Output: ~0.87 (Target: 1)  
Inputs: 1 0 → Output: ~0.87 (Target: 1)  
Inputs: 1 1 → Output: ~0.00 (Target: 0)  
```

---

## Getting Started

### Prerequisites
- A C++ compiler (e.g., `g++`)

### Compilation
```bash
g++ -o xor_nn xor_nn.cpp
```

### Run
```bash
./xor_nn
```

---

## File Structure
```
.
├── xor_nn.cpp   # Source code
├── README.md    # Project documentation
```

---

## License
This project is open-source and free to use for learning purposes.
