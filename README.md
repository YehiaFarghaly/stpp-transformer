# Spatio-Temporal Point Processing with Transformers

This project investigates Spatio-Temporal Point Processing using Transformer architectures, we explore two models:  

1 - Transformer with Traditional Neural Network (TNN): A conventional neural network model coupled with a Transformer for handling spatio-temporal point data.  
2 - Transformer with Kolmogorov-Arnold Network (KAN): A novel architecture integrating the Kolmogorov-Arnold representation theorem within a Transformer framework for more efficient learning of spatio-temporal dependencies.

Both models are implemented and benchmarked to study their performance on various tasks involving spatio-temporal data.


## Installation

To set up the environment and install the necessary libraries, you can use the provided `requirements.txt` file. You can set up the environment using `pip` as follows:

```bash
  pip install -r requirements.txt
```

To train the model we run the following script: 

```bash
bash run.sh train
```

You can modify the model architecture and experiment with different hyperparameters by editing the respective model scripts and configuration files in the repository

--- 
### Implementation Details could be found in the following paper

[Simulation And Inference Of Point Processes | Yehia Farghaly.pdf](https://github.com/user-attachments/files/17050439/Simulation.And.Inference.Of.Point.Processes.pdf)
