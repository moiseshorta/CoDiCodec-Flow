#!/usr/bin/env python3
"""
TUI monitoring component for training metrics.

This provides a terminal user interface to monitor training progress
in real-time without needing to tail log files.
"""
import subprocess
import re
import threading
import time
from pathlib import Path
from typing import Optional
import sys


class TrainingMonitor:
    """Monitors training process and displays metrics in TUI format."""
    
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self.metrics = {
            "step": 0,
            "loss": 0.0,
            "lr": 0.0,
            "val_loss": 0.0,
            "eta": "0:00:00",
        }
    
    def start_training(self, args: list) -> None:
        """Start the training process with given arguments."""
        cmd = [sys.executable, "-m", "flow.train"] + args
        print(f"Starting training: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        self.running = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_output)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start display thread
        display_thread = threading.Thread(target=self._display_metrics)
        display_thread.daemon = True
        display_thread.start()
        
        # Wait for process to complete
        self.process.wait()
        self.running = False
        
    def _monitor_output(self) -> None:
        """Monitor training process output and extract metrics."""
        if not self.process:
            return
            
        for line in self.process.stdout:
            if not self.running:
                break
                
            line = line.strip()
            
            # Parse training metrics from log output
            # Common patterns: step=1000 loss=0.5 lr=1e-4
            step_match = re.search(r'step[=:](\d+)', line)
            loss_match = re.search(r'loss[=:]([\d.]+)', line)
            lr_match = re.search(r'lr[=:]([\d.e-]+)', line)
            val_loss_match = re.search(r'val_loss[=:]([\d.]+)', line)
            
            if step_match:
                self.metrics["step"] = int(step_match.group(1))
            if loss_match:
                self.metrics["loss"] = float(loss_match.group(1))
            if lr_match:
                self.metrics["lr"] = float(lr_match.group(1))
            if val_loss_match:
                self.metrics["val_loss"] = float(val_loss_match.group(1))
            
            # Print raw output for debugging
            print(f"\r{line}", end="", flush=True)
    
    def _display_metrics(self) -> None:
        """Display training metrics in a clean TUI format."""
        last_update = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Update display every 0.5 seconds
            if current_time - last_update >= 0.5:
                self._clear_line()
                self._print_metrics()
                last_update = current_time
            
            time.sleep(0.1)
        
        # Final display
        self._clear_line()
        self._print_metrics()
        print()  # New line after training completes
    
    def _clear_line(self) -> None:
        """Clear the current line in terminal."""
        print("\r\033[K", end="", flush=True)
    
    def _print_metrics(self) -> None:
        """Print current training metrics."""
        step = self.metrics["step"]
        loss = self.metrics["loss"]
        lr = self.metrics["lr"]
        val_loss = self.metrics["val_loss"]
        
        # Format metrics
        metrics_str = (
            f"Step: {step:>8} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Val Loss: {val_loss:.4f}" if val_loss > 0 else
            f"Step: {step:>8} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e}"
        )
        
        print(f"\r{metrics_str}", end="", flush=True)


def launch_tui_train(
    data_dir: str,
    out_dir: str,
    device: str = "mps",
    batch_size: int = 8,
    grad_accum: int = 2,
    crop_tokens: int = 512,
    max_steps: int = 200000,
    dtype: str = "bf16",
    lr: float = 1e-4,
    dim: Optional[int] = None,
    n_layers: Optional[int] = None,
    n_heads: Optional[int] = None,
    cond_dim: Optional[int] = None,
) -> None:
    """Launch training with TUI monitoring."""
    # Build training arguments
    args = [
        "--data-dir", data_dir,
        "--out-dir", out_dir,
        "--device", device,
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--crop-tokens", str(crop_tokens),
        "--max-steps", str(max_steps),
        "--dtype", dtype,
    ]
    
    if dim:
        args.extend(["--dim", str(dim)])
    if n_layers:
        args.extend(["--n-layers", str(n_layers)])
    if n_heads:
        args.extend(["--n-heads", str(n_heads)])
    if cond_dim:
        args.extend(["--cond-dim", str(cond_dim)])
    
    # Create monitor and start training
    monitor = TrainingMonitor(out_dir)
    
    try:
        monitor.start_training(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if monitor.process:
            monitor.process.terminate()
    except Exception as e:
        print(f"\nError during training: {e}")
        if monitor.process:
            monitor.process.terminate()
        raise


if __name__ == "__main__":
    # Test the monitor
    print("TUI Monitor for codicodec-flow training")
    print("This module is intended to be used via the CLI wrapper")
