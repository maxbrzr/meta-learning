source .venv/bin/activate
python hydra/example/run.py --multirun hydra/launcher=submitit_local \
mlflow.tracking=False \
