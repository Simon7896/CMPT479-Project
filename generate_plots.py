#!/usr/bin/env python3
"""
Generate plots from saved logs and (optionally) a checkpoint.

Primary behavior (JSON-driven):
    - Reads training history from model/logs/training_results.json
    - Saves two separate figures:
            * training_loss.png       (train/val loss vs epoch)
            * training_accuracy.png   (train/val accuracy vs epoch)
    - Plots confusion_matrix.png if confusion_matrix is present in JSON
    - ROC curve requires FPR/TPR arrays; if not in JSON, can optionally compute
        via --eval-roc (loads checkpoint and evaluates test set to produce roc_curve.png)

Usage (inside container):
    # JSON-only (default)
    python3 generate_plots.py

    # Include ROC by running evaluation (needs data and checkpoint)
    python3 generate_plots.py --eval-roc --data_dir data/outputs --ckpt best

Outputs are saved under config.log_dir (default: ./model/logs/):
    - training_loss.png
    - training_accuracy.png
    - confusion_matrix.png (if available)
    - roc_curve.png (if available or computed with --eval-roc)
"""
import argparse
import os
import sys
import json
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from model import GCNConfig, GCNTrainer, prepare_data_loaders


def pick_checkpoint(dir_path: str, preference: str | None) -> str:
    if preference:
        cand = os.path.join(dir_path, f"{preference}_model.pth")
        if os.path.isfile(cand):
            return cand
    # Fallback order: final, best
    for name in ("final_model.pth", "best_model.pth"):
        cand = os.path.join(dir_path, name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"No checkpoint found in {dir_path}")


def _safe_json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_history_series(
    train: List[float],
    val: List[float],
    ylabel: str,
    title: str,
    save_path: str,
):
    plt.figure(figsize=(6, 4))
    plt.plot(train, label="Train")
    plt.plot(val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(cmatrix, class_names: List[str], save_path: str):
    import numpy as np

    cmatrix = np.array(cmatrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cmatrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate cells
    thresh = cmatrix.max() / 2.0 if cmatrix.size else 0
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax.text(
                j,
                i,
                f"{cmatrix[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cmatrix[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate result plots from checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/outputs",
                        help="Directory containing JSON graph files")
    parser.add_argument("--ckpt", type=str, default=None, choices=(None, "final", "best"),
                        help="Checkpoint preference: final or best (default: auto)")
    parser.add_argument("--eval-roc", action="store_true",
                        help="If set, compute ROC by evaluating the test set (requires data + checkpoint)")
    parser.add_argument("--log-json", type=str, default=None,
                        help="Path to training_results.json (defaults to <log_dir>/training_results.json)")
    args = parser.parse_args()

    # Build config and trainer
    config = GCNConfig()
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_path, exist_ok=True)

    # 1) Load training history from JSON and generate loss/accuracy plots
    log_json_path = (
        args.log_json
        if args.log_json
        else os.path.join(config.log_dir, "training_results.json")
    )
    if not os.path.isfile(log_json_path):
        print(f"Error: training history JSON not found at {log_json_path}")
        return 2
    logs = _safe_json_load(log_json_path)
    hist = (logs or {}).get("training_history", {})
    train_losses = hist.get("train_losses", [])
    val_losses = hist.get("val_losses", [])
    train_acc = hist.get("train_accuracies", [])
    val_acc = hist.get("val_accuracies", [])

    loss_out = os.path.join(config.log_dir, "training_loss.png")
    acc_out = os.path.join(config.log_dir, "training_accuracy.png")

    if train_losses and val_losses:
        _plot_history_series(
            train_losses, val_losses, "Loss", "Training and Validation Loss", loss_out
        )
        print(f"Saved: {loss_out}")
    else:
        print("Warning: Missing loss arrays in JSON; skipping training_loss.png")

    if train_acc and val_acc:
        _plot_history_series(
            train_acc, val_acc, "Accuracy", "Training and Validation Accuracy", acc_out
        )
        print(f"Saved: {acc_out}")
    else:
        print("Warning: Missing accuracy arrays in JSON; skipping training_accuracy.png")

    # 2) Confusion matrix: prefer JSON (aggregated) if available
    test_metrics = (logs or {}).get("test_metrics", {})
    cmatrix = test_metrics.get("confusion_matrix")
    if cmatrix is not None:
        cm_out = os.path.join(config.log_dir, "confusion_matrix.png")
        _plot_confusion_matrix(cmatrix, ["Non-vuln", "Vuln"], cm_out)
        print(f"Saved: {cm_out}")
    else:
        print("Note: No confusion_matrix in JSON; skipping confusion matrix plot.")

    # 3) ROC curve
    # If JSON has fpr/tpr, we could plot directly; otherwise optionally evaluate
    fpr = test_metrics.get("fpr")
    tpr = test_metrics.get("tpr")
    if fpr and tpr:
        from sklearn.metrics import auc as _auc

        roc_out = os.path.join(config.log_dir, "roc_curve.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {_auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Chance")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.savefig(roc_out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {roc_out}")
    elif args.eval_roc:
        # Evaluate to compute probs and draw ROC
        trainer = GCNTrainer(config)
        ckpt_path = pick_checkpoint(config.model_save_path, args.ckpt)
        print(f"Loading checkpoint for ROC eval: {ckpt_path}")
        trainer.load_model(os.path.basename(ckpt_path))

        print("Preparing data and evaluating test set for ROC...")
        _, _, test_loader = prepare_data_loaders(args.data_dir, config)
        _ = trainer.evaluate(test_loader)
        if trainer.last_eval and trainer.last_eval.get("probs"):
            roc_out = os.path.join(config.log_dir, "roc_curve.png")
            labels = list(trainer.last_eval.get("labels", []))
            probs = list(trainer.last_eval.get("probs", []))
            if labels and probs:
                trainer.plot_roc_curve(labels, probs, roc_out)
                print(f"Saved: {roc_out}")
            else:
                print("Warning: No ROC data available after eval.")
        else:
            print("Note: ROC curve not generated (requires binary probs).")
    else:
        print("Note: ROC curve not generated (no fpr/tpr in JSON; pass --eval-roc to compute).")

    print("Done. Plots saved under:", config.log_dir)
    for f in ("training_loss.png", "training_accuracy.png", "confusion_matrix.png", "roc_curve.png"):
        p = os.path.join(config.log_dir, f)
        status = "[OK]" if os.path.isfile(p) else "[SKIPPED]"
        print(" -", p, status)


if __name__ == "__main__":
    sys.exit(main())
