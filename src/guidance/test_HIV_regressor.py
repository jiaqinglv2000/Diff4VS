# Rdkit import should be first, do not move it
from rdkit import Chem

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings

import src.utils as utils
import src.datasets.hiv_dataset_a as hiv_dataset
from src.metrics.molecular_metrics import SamplingMolecularMetrics
from src.metrics.molecular_metrics_discrete import  TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.guidance.HIV_regressor_discrete import HIVRegressorDiscrete
from src.guidance.HIV_regressor_discrete_focal_loss import HIVRegressorDiscreteFocalLoss
from src.guidance.HIV_regressor_discrete_e_loss import HIVRegressorDiscreteELoss

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': 'graph_ddm_regressor', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.1', config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    print(cfg)
    dataset_config = cfg["dataset"]
    print(dataset_config)
    assert dataset_config["name"] == 'HIV'
    assert cfg.model.type == 'discrete'
    datamodule = hiv_dataset.HIVDataModule(cfg, regressor=True)
    dataset_infos = hiv_dataset.HIVinfos(datamodule=datamodule, cfg=cfg)

    datamodule.prepare_data()
    train_smiles = None

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
    #dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 2 if cfg.general.guidance_target == 'both' else 1}
    #dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 1}
    dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 1}
    print(dataset_infos.output_dims)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(remove_h = False, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    #model = HIVRegressorDiscreteELoss(cfg=cfg, **model_kwargs)
    model = HIVRegressorDiscreteELoss.load_from_checkpoint(
        '/home/acm512/DiGress/src/outputs/2024-04-05/11-33-45/checkpoints/regressor/epoch=1199.ckpt')
    #model = HIVRegressorDiscrete(cfg=cfg, **model_kwargs)
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_mae',
                                              save_last=True,
                                              save_top_k=-1,    # was 5
                                              mode='min',
                                              every_n_epochs=1)
        print("Checkpoints will be logged to", checkpoint_callback.dirpath)
        callbacks.append(checkpoint_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if cfg.general.gpus > 0 and torch.cuda.is_available() else 'cpu',
                      devices=1 if cfg.general.gpus > 0 and torch.cuda.is_available() else None,
                      limit_train_batches=20,     # TODO: remove
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks)


    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)


if __name__ == '__main__':
    main()


