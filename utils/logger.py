"""
Structured CSV Logger and ETA Engine for Snapshot Hyperspectral Imaging.

This module implements the structured CSV logging and dynamic rolling ETA engine
standardized in Yamawaki_Rep_2 and specified for Xie_Rep_2 in REVERSE_ENGINEERING_PLAN.md:

1. CSV Schema (Exact 8 Columns):
   Epoch,Train_Loss,Val_PSNR_dB,Val_SSIM,Val_SAM_rad,Learning_Rate,Epoch_Time_s,Estimated_Time_Remaining

2. Dynamic Rolling ETA Engine:
   Computes a rolling average of epoch execution durations over the past 10 epochs
   to filter timing jitter, projecting remaining time strictly formatted as:
       DD:HH:MM:SS (e.g. 00:01:23:45)

3. Fail-Safe Formatting:
   Handles non-finite, negative, or zero durations gracefully, returning "00:00:00:00".
"""

import os
import csv
from typing import Optional, Union, Dict, List, Any
import numpy as np


def format_eta(seconds: Union[float, int]) -> str:
    """
    Formats remaining seconds into days:hours:minutes:seconds (DD:HH:MM:SS).
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted ETA string 'DD:HH:MM:SS'. Returns '00:00:00:00' if seconds <= 0
        or non-finite.
    """
    if seconds is None:
        return "00:00:00:00"
    try:
        val = float(seconds)
    except (ValueError, TypeError):
        return "00:00:00:00"

    if val < 0 or not np.isfinite(val):
        return "00:00:00:00"

    secs = int(round(val))
    days = secs // 86400
    hours = (secs % 86400) // 3600
    minutes = (secs % 3600) // 60
    rem_secs = secs % 60
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{rem_secs:02d}"


class CSVLogger:
    """
    Structured CSV progress logger with dynamic rolling ETA estimation.
    
    Attributes:
        COLUMNS: Exact 8-column header list matching Yamawaki_Rep_2 and REVERSE_ENGINEERING_PLAN.md.
        log_path: Absolute or relative filesystem path to training_log.csv.
        epoch_durations: In-memory history of epoch execution times in seconds.
        history: Dictionary containing recorded metric series.
    """
    COLUMNS: List[str] = [
        "Epoch",
        "Train_Loss",
        "Val_PSNR_dB",
        "Val_SSIM",
        "Val_SAM_rad",
        "Learning_Rate",
        "Epoch_Time_s",
        "Estimated_Time_Remaining",
    ]

    def __init__(
        self,
        log_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Initializes the CSVLogger and prepares the destination log file.
        
        Args:
            log_path: Explicit path to CSV file. Defaults to 'results/training_log.csv'.
            results_dir: Optional directory containing 'training_log.csv'.
            overwrite: If True, truncates existing log and writes fresh header.
        """
        if log_path is not None:
            self.log_path = os.path.abspath(log_path)
        elif results_dir is not None:
            self.log_path = os.path.abspath(os.path.join(results_dir, "training_log.csv"))
        else:
            # Default relative to repository root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log_path = os.path.join(base_dir, "results", "training_log.csv")

        # Ensure parent directory exists
        parent_dir = os.path.dirname(self.log_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        self.epoch_durations: List[float] = []
        self.history: Dict[str, List[Any]] = {
            'epoch': [],
            'train_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'val_sam': [],
            'lr': [],
            'epoch_time': [],
            'eta': [],
        }

        # Initialize file with header
        if overwrite or not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)
        else:
            self._load_existing_durations()

    def _load_existing_durations(self) -> None:
        """Reads existing epoch durations and history from log file if present."""
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dur = float(row.get("Epoch_Time_s", 0.0))
                        self.epoch_durations.append(dur)
                        self.history['epoch'].append(int(row.get("Epoch", 0)))
                        self.history['train_loss'].append(float(row.get("Train_Loss", 0.0)))
                        self.history['val_psnr'].append(float(row.get("Val_PSNR_dB", 0.0)))
                        self.history['val_ssim'].append(float(row.get("Val_SSIM", 0.0)))
                        self.history['val_sam'].append(float(row.get("Val_SAM_rad", 0.0)))
                        self.history['lr'].append(float(row.get("Learning_Rate", 0.0)))
                        self.history['epoch_time'].append(dur)
                        self.history['eta'].append(str(row.get("Estimated_Time_Remaining", "00:00:00:00")))
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass

    @staticmethod
    def format_eta(seconds: Union[float, int]) -> str:
        """Convenience method delegating to module-level format_eta."""
        return format_eta(seconds)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_psnr: float,
        val_ssim: float,
        val_sam: float,
        lr: float,
        epoch_time: float,
        total_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculates rolling average duration of the last 10 epochs, estimates remaining time,
        appends a row to the CSV file, and updates in-memory history.
        
        Args:
            epoch: Current epoch index.
            train_loss: Training loss value.
            val_psnr: Validation PSNR in dB.
            val_ssim: Validation SSIM score in [0.0, 1.0].
            val_sam: Validation SAM score in radians.
            lr: Current optimizer learning rate.
            epoch_time: Execution duration for current epoch in seconds.
            total_epochs: Total number of planned epochs (for ETA calculation).
            
        Returns:
            Dictionary containing the formatted row values.
        """
        dur = float(epoch_time)
        self.epoch_durations.append(dur)

        # Rolling average over the past 10 epochs to prevent jitter
        recent_durations = self.epoch_durations[-10:]
        avg_epoch_time = float(np.mean(recent_durations)) if recent_durations else dur

        if total_epochs is not None and total_epochs > epoch:
            remaining_epochs = total_epochs - epoch
            remaining_seconds = avg_epoch_time * remaining_epochs
            eta_str = format_eta(remaining_seconds)
        else:
            eta_str = "00:00:00:00"

        # Formatted string representations per specification
        row_dict = {
            "Epoch": int(epoch),
            "Train_Loss": f"{float(train_loss):.6f}",
            "Val_PSNR_dB": f"{float(val_psnr):.2f}",
            "Val_SSIM": f"{float(val_ssim):.4f}",
            "Val_SAM_rad": f"{float(val_sam):.4f}",
            "Learning_Rate": f"{float(lr):.6f}",
            "Epoch_Time_s": f"{dur:.2f}",
            "Estimated_Time_Remaining": eta_str,
        }

        row_list = [
            row_dict["Epoch"],
            row_dict["Train_Loss"],
            row_dict["Val_PSNR_dB"],
            row_dict["Val_SSIM"],
            row_dict["Val_SAM_rad"],
            row_dict["Learning_Rate"],
            row_dict["Epoch_Time_s"],
            row_dict["Estimated_Time_Remaining"],
        ]

        # Atomically append row to CSV file
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_list)

        # Update in-memory history
        self.history['epoch'].append(int(epoch))
        self.history['train_loss'].append(float(train_loss))
        self.history['val_psnr'].append(float(val_psnr))
        self.history['val_ssim'].append(float(val_ssim))
        self.history['val_sam'].append(float(val_sam))
        self.history['lr'].append(float(lr))
        self.history['epoch_time'].append(dur)
        self.history['eta'].append(eta_str)

        return row_dict

    def get_history(self) -> Dict[str, List[Any]]:
        """Returns the dictionary of recorded metrics history."""
        return self.history

    def __len__(self) -> int:
        return len(self.history['epoch'])

    def __repr__(self) -> str:
        return f"CSVLogger(path='{self.log_path}', logged_epochs={len(self)})"


if __name__ == '__main__':
    # Standalone unit self-test
    print("Testing CSVLogger & format_eta...")
    assert format_eta(0) == "00:00:00:00"
    assert format_eta(90165) == "01:01:02:45"
    assert format_eta(-10) == "00:00:00:00"
    assert format_eta(float('nan')) == "00:00:00:00"
    assert format_eta(float('inf')) == "00:00:00:00"
    print("[+] format_eta tests passed.")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_csv = os.path.join(tmpdir, "test_log.csv")
        logger = CSVLogger(log_path=test_csv)
        assert os.path.exists(test_csv)
        res = logger.log_epoch(
            epoch=1,
            train_loss=0.0123456,
            val_psnr=32.456,
            val_ssim=0.89123,
            val_sam=0.12345,
            lr=0.001,
            epoch_time=5.25,
            total_epochs=100
        )
        assert res["Epoch"] == 1
        assert res["Train_Loss"] == "0.012346"
        assert res["Val_PSNR_dB"] == "32.46"
        assert res["Val_SSIM"] == "0.8912"
        assert res["Val_SAM_rad"] == "0.1235"
        assert res["Learning_Rate"] == "0.001000"
        assert res["Epoch_Time_s"] == "5.25"
        assert len(res["Estimated_Time_Remaining"].split(':')) == 4
        print(f"[+] Sample row: {res}")
    print("[+] All CSVLogger self-tests passed successfully.")
