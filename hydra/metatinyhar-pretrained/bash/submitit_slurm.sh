source .venv/bin/activate
export PROJECT_ROOT="$(pwd)"
python hydra/metatinyhar-pretrained/run.py --multirun hydra/launcher=custom_submitit_slurm \
mlflow.tracking=False \
