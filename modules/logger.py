import json
import wandb
from pathlib import Path

class FileLogger:
    def __init__(self, args):
        self.args = args
        self.log_dir = Path(self.args.checkpoint_dir)

    def log_epoch(self, log):
        with open(self.log_dir / 'run-logs.json', 'a') as f:
            f.write(f'{json.dumps(log)}\n')

    def log_table(self, dataframe):
        name = f'results-{self.args.visual_extractor}'
        dataframe.to_csv(self.log_dir / f'{name}.csv', index=False)

    def log_model(self, log, _):
        name = f'model-{self.args.visual_extractor}'
        with open(self.log_dir / f'{name}.json', 'a') as f:
            f.write(f'{json.dumps({"test/" + k: v for k, v in log.items()})}\n')


class WandbLogger(FileLogger):
    def __init__(self, args):
        super(WandbLogger, self).__init__(args)

    def log_epoch(self, log):
        super().log_epoch(log)
        log_dict = log.copy()
        epoch = log_dict.pop('epoch')
        wandb.log(log_dict, step=epoch)

    def log_table(self, dataframe):
        super().log_table(dataframe)
        name = f'results-{self.args.visual_extractor}'
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
        name = f'model-{self.args.visual_extractor}'
        artifact = wandb.Artifact(
            name,
            type='model',
            metadata=log
        )
        artifact.add_file(path_model)
        wandb.run.log_artifact(artifact)
