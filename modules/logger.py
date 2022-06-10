import json
import wandb
from pathlib import Path


class FileLogger:
    def __init__(self, log_dir, model_name):
        self.log_dir = Path(log_dir)
        self.model_name = model_name

    def log_epoch(self, log):
        with open(self.log_dir / 'run-logs.json', 'a') as f:
            f.write(f'{json.dumps(log)}\n')

    def log_table(self, dataframe):
        name = f'results-{self.model_name}'
        dataframe.to_csv(self.log_dir / f'{name}.csv', index=False)

    def log_model(self, log, _ = None):
        name = f'model-{self.model_name}'
        with open(self.log_dir / f'{name}.json', 'a') as f:
            f.write(f'{json.dumps({"test/" + k: v for k, v in log.items()})}\n')


class WandbLogger(FileLogger):
    def __init__(self, log_dir, model_name):
        super(WandbLogger, self).__init__(log_dir, model_name)

    def log_epoch(self, log):
        super().log_epoch(log)
        log_dict = log.copy()
        epoch = log_dict.pop('epoch')
        wandb.log(log_dict, step=epoch)

    def log_table(self, dataframe):
        super().log_table(dataframe)
        name = f'results-{self.model_name}'
        table = wandb.Table(dataframe=dataframe)
        table_artifact = wandb.Artifact(
            name,
            type='dataset'
        )
        table_artifact.add(table, name)
        table_artifact.add_file(self.log_dir / f'{name}.csv')

        wandb.log({name: table})
        wandb.run.log_artifact(table_artifact)

    def log_model(self, log, path_model):
        super().log_model(log)
        name = f'model-{self.model_name}'
        artifact = wandb.Artifact(
            name,
            type='model',
            metadata=log
        )
        artifact.add_file(self.log_dir / path_model)
        wandb.run.log_artifact(artifact)
