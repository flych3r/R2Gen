import json
import wandb
from pathlib import Path

class FileLogger:
    def __init__(self, args):
        self.args = args

    def log_epoch(self, log, path):
        with open(path, 'a') as f:
            f.write(f'{json.dumps(log)}\n')

    def log_table(self, dataframe, path):
        dataframe.to_csv(path, index=False)

    def log_model(self, log, path):
        with open(path, 'a') as f:
            f.write(f'{json.dumps({"test/" + k: v for k, v in log.items()})}\n')


class WandbLogger:
    def __init__(self, args):
        self.args = args

    def log_epoch(self, log, _):
        log_dict = log.copy()
        epoch = log_dict.pop('epoch')
        wandb.log(log_dict, step=epoch)

    def log_table(self, dataframe, name):
        name = Path(name).stem
        table = wandb.Table(dataframe=dataframe)
        wandb.log({name: table})

    def log_model(self, log, path):
        artifact = wandb.Artifact(
            f'model-{self.args.visual_extractor}',
            type='model',
            metadata=log
        )
        artifact.add_file(path)
        wandb.run.log_artifact(artifact)
