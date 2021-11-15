import hydra
from omegaconf import DictConfig
from hijax.setup import setup_worker

from neural_chess import MODULE_NAME


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    # get the experiment name
    name = cfg.get("name", False)
    if not name:
        raise Exception("Must specify experiment name on CLI. e.g. `python train.py name=vae ...`")

    # setup the worker
    overwrite = cfg.get("overwrite", False)
    reset_metrics = cfg.get("reset_metrics", False)
    worker, cfg = setup_worker(
        name=name,
        cfg=cfg,
        overwrite=overwrite,
        reset_metrics=reset_metrics,
        module=MODULE_NAME,
        with_wandb=True,
        with_loaders=True,
    )

    # train
    worker.run(nb_epoch=cfg["nb_epoch"])


if __name__ == "__main__":
    train()
