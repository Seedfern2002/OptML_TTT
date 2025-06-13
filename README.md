Very brief readme for now, add title
## give brief introduction
## how to get / download data
## how to train models?
## File Structure
- `experiments.py`: contains all main experiment functions.
- `run.ipynb`: a notebook that produces training and experiment results' (table and plots).
- `training.py`: Script for training models. It supports various training configurations via command-line arguments:
  - `--epochs`: Number of training epochs (default: `10`).
  - `--save_dir`: Directory to save training results and model checkpoints (default: `"results"`).
  - `--optimizer`: Choice of optimizer (`"adam"` or `"sgd"`, default: `"adam"`).
  - `--criterion`: Loss function for training (`"mse"`, `"cross_entropy"`, or `"kl_div"`, default: `"mse"`).
  - `--no_momentum`: Disable momentum when using SGD.
  - `--with_test`: Evaluate model on the test set during training.
  - `--disable_wandb`: Disable Weights & Biases logging.
  - `--log_file`: Enable logging of training progress to a file.
  - `--save_per_epoch`: Save the model at the end of each training epoch.
- `src/`: contains Python files such as data generation & loader, Tic-Tac-Toe, etc.
- `utils/`: contains Python files with helper functions for evaluation and plotting.
