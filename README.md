# FairBiNN: Fair Bilevel Neural Network

This repository contains the implementation of the paper:  
**Fair Bilevel Neural Network (FairBiNN): On Balancing Fairness and Accuracy via Stackelberg Equilibrium**  
Accepted at NeurIPS 2024.
### [paper on arXiv](https://arxiv.org/abs/2410.16432)

## Abstract

The persistent challenge of bias in machine learning models necessitates robust solutions to ensure parity and equal treatment across diverse groups, especially in classification tasks. Current bias mitigation techniques often result in information loss and inadequate balance between accuracy and fairness. To address this, we propose a novel methodology based on bilevel optimization principles. Our deep learning-based approach optimizes for both accuracy and fairness objectives simultaneously, achieving proven Pareto optimal solutions while mitigating bias in the trained model. Theoretical analysis shows that the upper bound on loss incurred by this method is less than or equal to that of the Lagrangian approach. We demonstrate the efficacy of FairBiNN primarily on tabular datasets, such as UCI Adult and Heritage Health. It outperforms state-of-the-art fairness methods, advancing fairness-aware machine learning solutions by effectively bridging the accuracy-fairness gap.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Parameters](#parameters)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fairbinn.git
   cd fairbinn
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, use the following command:

```bash
python main.py --mode [fairness_mode] --eta [eta_value] --data [dataset_name] --fairness_layers [fairness_layers] --acc_layers [accuracy_layers] --fairness_position [fairness_position] --device [device] --epochs [num_epochs] --batch_size [batch_size]
```

### Parameters

- `--mode`: Fairness layer mode. Options: `reg`, `adv`, `hybrid` (default: `reg`).
- `--eta`: Regularization parameter for fairness loss (default: 10000).
- `--data`: Dataset to use. Options: `Adult`, `Health` (default: `Adult`).
- `--fairness_layers`: Number of hidden units in the fairness layers (default: based on dataset).
- `--acc_layers`: Number of hidden units in the accuracy layers (default: based on dataset).
- `--fairness_position`: Position of fairness layers in the network (default: 2).
- `--device`: Device for training. Options: `cpu`, `cuda`, `mps` (default: `cpu`).
- `--epochs`: Number of training epochs (default: 200).
- `--batch_size`: Batch size for training (default: 200).

### Example

```bash
python main.py --mode adv --eta 5000 --data Adult --fairness_layers 128 128 --acc_layers 128 128 1 --fairness_position 2 --device cuda --epochs 100 --batch_size 128
```

## Results

The performance of FairBiNN has been tested on the UCI Adult and Heritage Health datasets, demonstrating superior fairness and accuracy trade-offs compared to state-of-the-art methods.

<!-- ## Citation

If you find this work useful, please cite our paper:

```
@inproceedings{fairbinn2024,
  title={Fair Bilevel Neural Network (FairBiNN): On Balancing Fairness and Accuracy via Stackelberg Equilibrium},
  author={Mehdi Yazdani-Jahromi, Ali Khodabandeh Yalabadi, Aida Tayebi, Amirarsalan Rajabi, Ivan Garibay, Ozlem Ozmen Garibay},
  booktitle={NeurIPS},
  year={2024},
  url={PLACEHOLDER_FOR_LINK}
}
``` -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
