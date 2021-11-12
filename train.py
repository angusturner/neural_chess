import hydra
from omegaconf import DictConfig
from hijax.setup import setup_worker, setup_loaders

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
        overwrite=overwrite,
        reset_metrics=reset_metrics,
        module=MODULE_NAME,
    )

    # setup data loaders
    dataset_class = cfg["dataset"].pop("class")
    train_loader, test_loader = setup_loaders(
        dataset_class=dataset_class, data_opts=cfg["dataset"], loader_opts=cfg["loader"], module=MODULE_NAME
    )

    # train
    worker.run(train_loader, test_loader, cfg["nb_epoch"])


if __name__ == "__main__":
    train()
