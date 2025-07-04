# TTTTT: The Impact of Training Order in Tic-Tac-Toe
## Introduction
This project explores how the **order of training samples** affects model performance in the game of **Tic-Tac-Toe**. We use **Monte Carlo Tree Search (MCTS)** to generate expert gameplay data and train a **Convolutional Neural Network (CNN)** to simulate the behavior of the MCTS agent.

From a human perspective, game states with **more stones on the board** are often easier to evaluate, because there are fewer legal moves and the decision tree is shallower. This observation raises an interesting question:

> *Does the order of training sample—starting from easy to hard, hard to easy, or random—affect the learning outcome of the model?*

Let’s find out how training order shapes learning in Tic-Tac-Toe!

## File Structure

- `src/`: Core source code for data generation, game logic, model training, and evaluation.
  - `tictactoe.py`: Implementation of the Tic-Tac-Toe game environment.
  - `mcts.py`: Runs the MCTS agent to generate gameplay data.
  - `data_generator.py`: Uses self-play with MCTS to generate training data.
  - `dataloader.py`: Loads training and test datasets.
  - `train.py`: Training logic used by `training.py`.
  - `eval.py`: Evaluation functions for assessing model performance.
  
- `model/`:
  - `model.py`: Defines the CNN model architecture.

- `training.py`: Script for training models, with configurable hyperparameters.

- `experiments.py`: Contains core experiment functions for running and comparing different settings.

- `run.ipynb`: Jupyter notebook for generating plots and tables from training and evaluation results. The first few cells are for training and running experiments, therefore it produces a lot of logs. To check the resulting plots and numbers, we advise you to check the last three cells. The entire run will take around 30 minutes.

- `utils/`: Utility functions for evaluation and visualization.
  - `eval_utils.py`: Helpers for computing evaluation metrics.
  - `plot_utils.py`: Helpers for plotting experiment results.


## Install the dependency

Install the required dependencies using:

```bash
conda env create --file requirements.yml
```

## Getting the Data

You can either **generate the data yourself** or **download it directly** (recommended).

### Option 1: Download the data
You can download the dataset from the following link:  
[Google Drive Folder](https://drive.google.com/drive/folders/1Nh7CXp5Gk3135Za5Cj1uRTLYP6VGn4Y3?usp=sharing)

The folder contains the following files:

- **`monte_carlo_data.zip`**: Includes training data in the form of `(state, MCTS results)` pairs used to train the model.
- **`symmetries.zip`**: Contains mappings of each game state to its symmetrical variants. This is used to ensure a proper train-test split by accounting for symmetrical duplicates.

After downloading, unzip the files and place the folders `monte_carlo_data/` and `symmetries/` in the **root directory** of the experiment.

### Option 2: Generate the Data

Run the following command:

```bash
python -m src.data_generator
```

## Getting Results

To reproduce the results presented in the report, simply open and run the `run.ipynb` notebook.

> **Note:** Although random seeds are fixed, you may not obtain *exactly* the same results as shown in the report due to sources of randomness in PyTorch, differences in operating systems, or using different devices (e.g., CPU vs. GPU).
> 
> See PyTorch's [reproducibility guidelines](https://pytorch.org/docs/stable/notes/randomness.html) for more information.
> 
> However, the overall trends and visualizations should remain similar, as the training process is generally stable.

## Train the Model with Custom Hyperparameters

To train the model with your own configuration, run:

```bash
python -m training
```
It will run training with easy-to-hard, hard-to-easy, random orders and save the corresponding loss, competition results.

You can customize the training process using the following command-line arguments:
  - `--epochs`: Number of training epochs (default: `10`).
  - `--save_dir`: Directory to save training results and model checkpoints (default: `"results"`).
  - `--optimizer`: Choice of optimizer (`"adam"` or `"sgd"`, default: `"adam"`).
  - `--criterion`: Loss function for training (`"mse"`, `"cross_entropy"`, or `"kl_div"`, default: `"mse"`).
  - `--no_momentum`: Disable momentum when using SGD.
  - `--with_test`: Evaluate model on the test set during training.
  - `--disable_wandb`: Disable Weights & Biases logging.
  - `--log_file`: Enable logging of training progress to a file.
  - `--save_per_epoch`: Save the model at the end of each training epoch.
