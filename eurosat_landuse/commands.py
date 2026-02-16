import json
import subprocess
from pathlib import Path

import fire
import mlflow
import mlflow.pyfunc
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from eurosat_landuse.data.datamodule import EuroSATDataConfig, EuroSATDataModule
from eurosat_landuse.data.download import download_eurosat_rgb_if_needed
from eurosat_landuse.infer.predictor import Predictor, PredictorConfig
from eurosat_landuse.models.factory import create_model
from eurosat_landuse.serving.mlflow_pyfunc import EuroSATPyFuncModel, ServingConfig
from eurosat_landuse.training.lit_module import LanduseLitModule
from eurosat_landuse.utils.git import get_git_commit_id
from eurosat_landuse.utils.plots import save_curves
from eurosat_landuse.utils.repro import seed_everything


def _load_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = Path.cwd() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _try_dvc_pull() -> None:
    try:
        subprocess.run(["dvc", "pull"], check=False, capture_output=True, text=True)
    except Exception:
        return


def _ensure_data(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data.data_dir)
    if data_dir.exists() and any(data_dir.glob("*/*.jpg")):
        return

    _try_dvc_pull()

    if data_dir.exists() and any(data_dir.glob("*/*.jpg")):
        return

    if not bool(cfg.data.download.enabled):
        raise FileNotFoundError(f"Data not found at {data_dir} and download disabled")

    download_eurosat_rgb_if_needed(
        data_dir=data_dir,
        url=str(cfg.data.download.url),
        zip_path=Path(cfg.data.download.zip_path),
        extract_to=Path(cfg.data.download.extract_to),
    )


def _extract_backbone_state_dict(lightning_ckpt_path: Path) -> dict:
    ckpt = torch.load(lightning_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            backbone_state[k.replace("model.", "", 1)] = v
    return backbone_state


class Commands:
    def download_data(self, *overrides: str) -> None:
        cfg = _load_cfg("train.yaml", list(overrides))
        _ensure_data(cfg)
        print(f"Data is ready at: {cfg.data.data_dir}")

    def train(self, *overrides: str) -> None:
        cfg = _load_cfg("train.yaml", list(overrides))
        print(OmegaConf.to_yaml(cfg))

        seed_everything(int(cfg.seed))
        _ensure_data(cfg)

        artifacts_dir = Path(cfg.artifacts_dir)
        plots_dir = Path(cfg.plots_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        datacfg = EuroSATDataConfig(
            data_dir=Path(cfg.data.data_dir),
            image_size=int(cfg.model.image_size),
            seed=int(cfg.data.split.seed),
            train_ratio=float(cfg.data.split.train_ratio),
            val_ratio=float(cfg.data.split.val_ratio),
            test_ratio=float(cfg.data.split.test_ratio),
            batch_size=int(cfg.train.batch_size),
            num_workers=int(cfg.data.dataloader.num_workers),
            pin_memory=bool(cfg.data.dataloader.pin_memory),
        )
        dm = EuroSATDataModule(datacfg)
        dm.setup()
        class_names = dm.class_names

        model = create_model(
            name=str(cfg.model.name),
            num_classes=int(cfg.model.num_classes),
            pretrained=bool(getattr(cfg.model, "pretrained", False)),
        )

        lit = LanduseLitModule(
            model=model,
            num_classes=int(cfg.model.num_classes),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
            class_names=class_names,
            plots_dir=plots_dir,
        )

        mlf_logger = MLFlowLogger(
            tracking_uri=str(cfg.logger.tracking_uri),
            experiment_name=str(cfg.logger.experiment_name),
            run_name=str(cfg.logger.run_name),
        )

        mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        mlf_logger.experiment.log_param(mlf_logger.run_id, "git_commit_id", get_git_commit_id())

        ckpt_callback = ModelCheckpoint(
            dirpath=str(artifacts_dir),
            filename="best",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=1,
        )

        trainer = pl.Trainer(
            max_epochs=int(cfg.trainer.max_epochs),
            accelerator=str(cfg.trainer.accelerator),
            devices=cfg.trainer.devices,
            precision=int(cfg.trainer.precision),
            log_every_n_steps=int(cfg.trainer.log_every_n_steps),
            logger=mlf_logger,
            callbacks=[
                ckpt_callback,
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(monitor="val/f1_macro", mode="max", patience=3),
            ],
        )

        trainer.fit(lit, datamodule=dm)
        trainer.test(lit, datamodule=dm, ckpt_path="best")

        curves = lit.epoch_curves
        epochs = range(1, len(curves["val_acc"]) + 1)
        save_curves(
            plots_dir=plots_dir,
            epochs=epochs,
            train_loss=curves["train_loss"],
            val_loss=curves["val_loss"],
            val_acc=curves["val_acc"],
        )

        best_ckpt_path = Path(ckpt_callback.best_model_path)
        final_ckpt_path = artifacts_dir / "best.ckpt"
        if best_ckpt_path.exists():
            final_ckpt_path.write_bytes(best_ckpt_path.read_bytes())

        backbone_state = _extract_backbone_state_dict(final_ckpt_path)
        weights_path = artifacts_dir / "model_weights.pth"
        torch.save(backbone_state, weights_path)

        if bool(cfg.export.onnx.enabled):
            self.export_onnx(
                f"model.name={cfg.model.name}",
                f"model.num_classes={cfg.model.num_classes}",
                f"model.image_size={cfg.model.image_size}",
                f"infer.checkpoint_path={final_ckpt_path.as_posix()}",
                f"export.onnx.path={cfg.export.onnx.path}",
                f"export.onnx.opset={cfg.export.onnx.opset}",
            )

        if bool(cfg.export.mlflow_model.enabled):
            self.export_mlflow_model(
                f"data.data_dir={cfg.data.data_dir}",
                f"model.name={cfg.model.name}",
                f"model.num_classes={cfg.model.num_classes}",
                f"model.image_size={cfg.model.image_size}",
                f"export.mlflow_model.path={cfg.export.mlflow_model.path}",
                f"artifacts_dir={cfg.artifacts_dir}",
            )

        mlflow.set_tracking_uri(str(cfg.logger.tracking_uri))
        with mlflow.start_run(run_id=mlf_logger.run_id):
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="artifacts")

        print(f"Checkpoint saved to: {final_ckpt_path}")
        print(f"Backbone weights saved to: {weights_path}")

    def export_onnx(self, *overrides: str) -> None:
        cfg = _load_cfg("train.yaml", list(overrides))

        ckpt_path = (
            Path(cfg.infer.checkpoint_path) if "infer" in cfg else Path("artifacts/best.ckpt")
        )
        onnx_path = Path(cfg.export.onnx.path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        model = create_model(
            name=str(cfg.model.name),
            num_classes=int(cfg.model.num_classes),
            pretrained=False,
        )

        backbone_state = _extract_backbone_state_dict(ckpt_path)
        model.load_state_dict(backbone_state, strict=True)
        model.eval()

        dummy = torch.randn(1, 3, int(cfg.model.image_size), int(cfg.model.image_size))
        torch.onnx.export(
            model,
            dummy,
            onnx_path.as_posix(),
            input_names=["image"],
            output_names=["logits"],
            opset_version=int(cfg.export.onnx.opset),
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )

        meta = {
            "model": str(cfg.model.name),
            "num_classes": int(cfg.model.num_classes),
            "image_size": int(cfg.model.image_size),
        }
        (onnx_path.parent / "onnx_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        print(f"Exported ONNX to: {onnx_path}")

    def export_mlflow_model(self, *overrides: str) -> None:
        cfg = _load_cfg("train.yaml", list(overrides))

        artifacts_dir = Path(cfg.artifacts_dir)
        weights_path = artifacts_dir / "model_weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        export_path = Path(cfg.export.mlflow_model.path)
        export_path.mkdir(parents=True, exist_ok=True)

        pyfunc_model = EuroSATPyFuncModel(
            cfg=ServingConfig(
                data_dir=Path(cfg.data.data_dir),
                model_name=str(cfg.model.name),
                num_classes=int(cfg.model.num_classes),
                image_size=int(cfg.model.image_size),
            )
        )

        mlflow.pyfunc.save_model(
            path=str(export_path),
            python_model=pyfunc_model,
            artifacts={"weights": str(weights_path)},
        )

        print(f"Saved MLflow model to: {export_path}")

    def infer(self, *overrides: str) -> None:
        cfg = _load_cfg("infer.yaml", list(overrides))
        print(OmegaConf.to_yaml(cfg))

        seed_everything(int(cfg.seed))
        _ensure_data(cfg)

        image_path_value = cfg.infer.image_path
        if image_path_value is None:
            raise ValueError("Set infer.image_path=/path/to/image.jpg")

        ckpt_path = Path(cfg.infer.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        from torchvision import datasets

        ds = datasets.ImageFolder(root=str(cfg.data.data_dir))
        class_names = list(ds.classes)

        model = create_model(
            name=str(cfg.model.name),
            num_classes=int(cfg.model.num_classes),
            pretrained=False,
        )

        backbone_state = _extract_backbone_state_dict(ckpt_path)
        model.load_state_dict(backbone_state, strict=True)

        predictor = Predictor(
            model=model,
            class_names=class_names,
            cfg=PredictorConfig(image_size=int(cfg.model.image_size), topk=int(cfg.infer.topk)),
        )

        result = predictor.predict(Path(str(image_path_value)))
        print(json.dumps(result, indent=2, ensure_ascii=False))

    def infer_mlflow_model(
        self, image_path: str, model_path: str = "artifacts/mlflow_model"
    ) -> None:
        model = mlflow.pyfunc.load_model(model_path)
        df = pd.DataFrame([{"image_path": image_path}])
        pred = model.predict(df)
        print(pred.to_json(orient="records", force_ascii=False, indent=2))


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
