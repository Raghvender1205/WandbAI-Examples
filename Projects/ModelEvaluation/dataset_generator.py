# Generate Dataset using Args
import wandb
import util
import argparse

project = 'model_registry_wandb_example'
model_use_case_id = 'mnist'
job_type = 'dataset_builder'

parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type=int, default=100, help='Number of Training Images')
run = wandb.init(project=project, job_type=job_type, config=parser.parse_args())

# Generate Raw Data
(X_train, y_train), (X_eval, y_eval) = util.generate_raw_data(run.config.train_size)

# Publish Dataset to Wandb
util.publish_dataset_to_wb(X_train, y_train, X_eval, y_eval, model_use_case_id)