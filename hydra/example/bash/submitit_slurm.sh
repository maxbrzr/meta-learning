source .venv/bin/activate
python hydra/example/run.py --multirun hydra/launcher=custom_submitit_slurm \
mlflow.tracking=False \
