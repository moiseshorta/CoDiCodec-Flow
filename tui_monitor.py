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
    
    def __init__(self, out_dir: str, max_steps: int = 200000):
        self.out_dir = Path(out_dir)
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self.max_steps = max_steps
        self.metrics = {
            "step": 0,
            "loss": 0.0,
            "lr": 0.0,
            "val_loss": 0.0,
            "eta": "0:00:00",
            "steps_per_sec": 0.0,
        }
        self.phase = "initializing"  # Can be: initializing, audio_generation, training, saving_checkpoint
        self.audio_generation_info = {
            "current_idx": 0,
            "total_samples": 2,
            "current_step": 0,
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
            # Format: step=686100/10000000  loss=0.8634  lr=9.94e-05  t=0.583  prefix=336/512  steps/s=2.39
            # Audio generation: [audio] step=686000 idx=0 (uncond) -> ./runs/v3_okachihuali/samples/step_0686000_idx_00.wav (12.29s)
            # Checkpoint saving: [checkpoint] step=686000 -> ./runs/v3_okachihuali/last.pt
            
            # Detect audio generation phase
            if "[audio]" in line:
                self.phase = "audio_generation"
                audio_idx_match = re.search(r'idx[=:](\d+)', line)
                audio_step_match = re.search(r'step[=:](\d+)', line)
                if audio_idx_match:
                    self.audio_generation_info["current_idx"] = int(audio_idx_match.group(1))
                if audio_step_match:
                    self.audio_generation_info["current_step"] = int(audio_step_match.group(1))
            elif "checkpoint" in line.lower() or "saving" in line.lower():
                self.phase = "saving_checkpoint"
                # Try to extract step from checkpoint line if available
                step_match = re.search(r'step[=:](\d+)', line)
                if step_match:
                    self.metrics["step"] = int(step_match.group(1))
            elif "step=" in line and "loss=" in line:
                self.phase = "training"
            
            step_match = re.search(r'step[=:](\d+)/\d+', line)
            loss_match = re.search(r'loss[=:]([\d.]+)', line)
            lr_match = re.search(r'lr[=:]([\d.e-]+)', line)
            val_loss_match = re.search(r'val_loss[=:]([\d.]+)', line)
            steps_per_sec_match = re.search(r'steps/s[=:]([\d.]+)', line)
            
            if step_match:
                self.metrics["step"] = int(step_match.group(1))
            if loss_match:
                self.metrics["loss"] = float(loss_match.group(1))
            if lr_match:
                self.metrics["lr"] = float(lr_match.group(1))
            if val_loss_match:
                self.metrics["val_loss"] = float(val_loss_match.group(1))
            if steps_per_sec_match:
                self.metrics["steps_per_sec"] = float(steps_per_sec_match.group(1))
    
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
        """Print current training metrics with progress bar."""
        if self.phase == "audio_generation":
            self._print_audio_generation()
        elif self.phase == "training":
            self._print_training_progress()
        elif self.phase == "saving_checkpoint":
            self._print_saving_checkpoint()
        else:
            self._print_initializing()
    
    def _print_saving_checkpoint(self) -> None:
        """Print checkpoint saving status."""
        step = self.metrics["step"]
        loading_chars = ["◐", "◑", "◒", "◔"]
        loading_idx = int(time.time() * 4) % 4
        loading_char = loading_chars[loading_idx]
        print(f"\r[{loading_char}] Saving checkpoint at step {step}...", end="", flush=True)
    
    def _print_audio_generation(self) -> None:
        """Print audio generation progress."""
        current_idx = self.audio_generation_info["current_idx"]
        current_step = self.audio_generation_info["current_step"]
        
        # Use unicode loading animation
        loading_chars = ["◐", "◑", "◒", "◔"]
        loading_idx = int(time.time() * 4) % 4
        loading_char = loading_chars[loading_idx]
        
        progress_str = f"\r[{loading_char}] Generating audio samples | Sample: {current_idx}/2 | Step: {current_step} | Initializing training..."
        print(f"{progress_str}", end="", flush=True)
    
    def _print_initializing(self) -> None:
        """Print initialization status."""
        loading_chars = ["◐", "◑", "◒", "◔"]
        loading_idx = int(time.time() * 4) % 4
        loading_char = loading_chars[loading_idx]
        print(f"\r[{loading_char}] Initializing training...", end="", flush=True)
    
    def _print_training_progress(self) -> None:
        """Print training metrics with progress bar."""
        step = self.metrics["step"]
        loss = self.metrics["loss"]
        lr = self.metrics["lr"]
        val_loss = self.metrics["val_loss"]
        steps_per_sec = self.metrics["steps_per_sec"]
        
        # Calculate progress
        progress = step / self.max_steps if self.max_steps > 0 else 0
        progress_percent = min(100.0, progress * 100)
        
        # Create progress bar with unicode gradient characters
        bar_width = 40
        filled_width = int(bar_width * progress)
        remainder = (bar_width * progress) - filled_width
        
        # Unicode gradient characters from light to dark
        gradient_chars = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
        
        # Determine which character to use for the partial fill
        if remainder > 0:
            char_idx = int(remainder * len(gradient_chars))
            partial_char = gradient_chars[min(char_idx, len(gradient_chars) - 1)]
        else:
            partial_char = ""
        
        bar = "█" * filled_width + partial_char + " " * (bar_width - filled_width - (1 if partial_char else 0))
        bar = bar[:bar_width]  # Ensure exact width
        
        # Calculate ETA if we have steps/sec
        eta_str = "N/A"
        if steps_per_sec > 0:
            remaining_steps = self.max_steps - step
            eta_seconds = remaining_steps / steps_per_sec
            eta_str = self._format_time(eta_seconds)
        
        # Format metrics
        metrics_str = (
            f"\r[{bar}] {progress_percent:>5.1f}% | "
            f"Step: {step:>8}/{self.max_steps} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Speed: {steps_per_sec:.2f} steps/s | "
            f"ETA: {eta_str}"
        )
        
        print(f"{metrics_str}", end="", flush=True)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def launch_tui_train(
    data_dir: str,
    out_dir: str,
    device: str = "mps",
    batch_size: int = 8,
    grad_accum: int = 2,
    crop_tokens: int = 512,
    max_steps: int = 200000,
    warmup_steps: int = 2000,
    dtype: str = "bf16",
    lr: float = 1e-4,
    num_workers: int = 0,
    log_every: int = 50,
    val_every: int = 1000,
    ckpt_every: int = 1000,
    audio_sample_every: int = 0,
    audio_n_samples: int = 2,
    audio_prompt_seconds: float = 4,
    audio_continuation_seconds: float = 8,
    audio_nfe: int = 16,
    audio_solver: str = "heun",
    audio_unconditional: bool = False,
    t_sample_mode: str = "uniform",
    dropout: float = 0.1,
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
        "--warmup-steps", str(warmup_steps),
        "--dtype", dtype,
        "--lr", str(lr),
        "--num-workers", str(num_workers),
        "--log-every", str(log_every),
        "--val-every", str(val_every),
        "--ckpt-every", str(ckpt_every),
        "--audio-sample-every", str(audio_sample_every),
        "--audio-n-samples", str(audio_n_samples),
        "--audio-prompt-seconds", str(audio_prompt_seconds),
        "--audio-continuation-seconds", str(audio_continuation_seconds),
        "--audio-nfe", str(audio_nfe),
        "--audio-solver", audio_solver,
        "--t-sample-mode", t_sample_mode,
        "--dropout", str(dropout),
    ]
    
    if audio_unconditional:
        args.append("--audio-unconditional")
    if dim:
        args.extend(["--dim", str(dim)])
    if n_layers:
        args.extend(["--n-layers", str(n_layers)])
    if n_heads:
        args.extend(["--n-heads", str(n_heads)])
    if cond_dim:
        args.extend(["--cond-dim", str(cond_dim)])
    
    # Create monitor and start training
    monitor = TrainingMonitor(out_dir, max_steps=max_steps)
    
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
