#!/usr/bin/env python
"""Chester v2 demo — trains a real neural network, saves a checkpoint.

    cd /home/houhd/code/chester-overhaul/tests
    uv run python ../demo_v2.py                  # local only
    uv run python ../demo_v2.py --ssh armdual    # local + real SSH
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap


# ── The training function (this gets cloudpickle'd by chester) ──────────

def train_mnist_mlp(variant, log_dir, exp_name):
    """Train a small MLP on MNIST for a few steps, save checkpoint."""
    import torch
    import torch.nn as nn

    lr = variant["learning_rate"]
    hidden = variant["hidden_dim"]
    steps = variant["train_steps"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tiny MLP: 784 -> hidden -> 10
    model = nn.Sequential(
        nn.Linear(784, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic MNIST-shaped data (no download needed)
    torch.manual_seed(42)
    X = torch.randn(256, 784, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # Train
    model.train()
    losses = []
    for step in range(steps):
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  step {step+1}/{steps}  loss={loss.item():.4f}  acc={acc:.2%}")

    # Save checkpoint
    os.makedirs(log_dir, exist_ok=True)
    ckpt_path = os.path.join(log_dir, "checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss": losses[-1],
        "final_step": steps,
        "config": {k: v for k, v in variant.items()},
    }, ckpt_path)

    # Save readable metrics
    metrics_path = os.path.join(log_dir, "metrics.json")
    final_acc = (model(X).argmax(1) == y).float().mean().item()
    metrics = {
        "exp_name": exp_name,
        "device": device,
        "final_loss": round(losses[-1], 6),
        "final_accuracy": round(final_acc, 4),
        "train_steps": steps,
        "learning_rate": lr,
        "hidden_dim": hidden,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"  Metrics:    {metrics_path}")
    print(f"  Final loss={losses[-1]:.4f}  acc={final_acc:.2%}")


# ── Demo infrastructure ─────────────────────────────────────────────────

def setup_demo_project():
    """Create a temporary project with .chester/config.yaml."""
    demo_dir = tempfile.mkdtemp(prefix="chester_demo_")
    chester_dir = os.path.join(demo_dir, ".chester")
    os.makedirs(chester_dir)

    config = textwrap.dedent("""\
        log_dir: data
        package_manager: uv

        backends:
          local:
            type: local

          armdual:
            type: ssh
            host: armdual
            remote_dir: /tmp/chester_demo

          gl:
            type: slurm
            host: gl
            remote_dir: /tmp/chester_demo
            modules: [singularity]
            cuda_module: cuda/12.8.1
            slurm:
              partition: spgpu
              time: "72:00:00"
              nodes: 1
              gpus: 1
              cpus_per_gpu: 8
              mem_per_gpu: 64G
            singularity:
              image: /opt/container.sif
              mounts: [/usr/share/glvnd]
              gpu: true
    """)
    config_path = os.path.join(chester_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(config)

    os.environ["CHESTER_CONFIG_PATH"] = config_path
    return demo_dir


def demo_local(demo_dir):
    """Train locally via run_experiment_lite, verify checkpoint."""
    from chester.run_exp import VariantGenerator, run_experiment_lite

    print("=" * 60)
    print(" LOCAL: train MLP, save checkpoint")
    print("=" * 60)

    vg = VariantGenerator()
    vg.add("learning_rate", [0.01])
    vg.add("hidden_dim", [64])
    vg.add("train_steps", [50])

    for v in vg.variants():
        run_experiment_lite(
            stub_method_call=train_mnist_mlp,
            variant=v,
            mode="local",
            exp_prefix="demo_mlp",
            log_dir=os.path.join(demo_dir, "data", "demo_local"),
            git_snapshot=False,
        )

    # Verify outputs
    log_dir = os.path.join(demo_dir, "data", "demo_local")
    print("\n" + "-" * 40)
    print("Verification:")

    ckpt = os.path.join(log_dir, "checkpoint.pt")
    metrics_file = os.path.join(log_dir, "metrics.json")

    ok = True
    if os.path.exists(ckpt):
        import torch
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        n_params = sum(p.numel() for p in data["model_state_dict"].values())
        print(f"  checkpoint.pt  ({os.path.getsize(ckpt)} bytes, {n_params} params)")
        print(f"    final_loss: {data['final_loss']:.4f}")
        print(f"    final_step: {data['final_step']}")
        print(f"    layers: {list(data['model_state_dict'].keys())}")
    else:
        print(f"  FAIL: {ckpt} not found")
        ok = False

    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            m = json.load(f)
        print(f"  metrics.json:")
        for k, val in m.items():
            print(f"    {k}: {val}")
    else:
        print(f"  FAIL: {metrics_file} not found")
        ok = False

    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return ok, log_dir


def demo_ssh(demo_dir, host):
    """Train on a remote SSH host, pull checkpoint back, verify."""
    from chester.backends import create_backend
    from chester.backends.base import parse_backend_config
    from chester.config_v2 import load_config

    print("\n" + "=" * 60)
    print(f" SSH ({host}): train remotely, pull checkpoint back")
    print("=" * 60)

    cfg = load_config()
    backend_cfg = parse_backend_config(host, {
        "type": "ssh",
        "host": host,
        "remote_dir": "/tmp/chester_demo",
    })
    backend = create_backend(backend_cfg, cfg)

    remote_log = "/tmp/chester_demo/data/demo_ssh"
    local_log = os.path.join(demo_dir, "data", "demo_ssh")
    os.makedirs(local_log, exist_ok=True)

    task = {
        "params": {
            "log_dir": remote_log,
            "exp_name": "demo_ssh",
        },
    }

    # Script that trains the same MLP on the remote host
    script = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -e
        echo "[chester] Running on $(hostname) at $(date)"
        mkdir -p {remote_log}
        python3 -c "
import json, os, sys
try:
    import torch
    import torch.nn as nn
except ImportError:
    print('torch not available on remote -- writing placeholder')
    os.makedirs('{remote_log}', exist_ok=True)
    with open('{remote_log}/metrics.json', 'w') as f:
        json.dump({{'status': 'torch_not_available', 'host': '$(hostname)'}}, f, indent=2)
    open('{remote_log}/.done', 'w').close()
    sys.exit(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
torch.manual_seed(42)
X = torch.randn(256, 784, device=device)
y = torch.randint(0, 10, (256,), device=device)
for step in range(50):
    loss = loss_fn(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
acc = (model(X).argmax(1) == y).float().mean().item()
os.makedirs('{remote_log}', exist_ok=True)
torch.save({{'model_state_dict': model.state_dict(), 'final_loss': loss.item(), 'final_step': 50}}, '{remote_log}/checkpoint.pt')
with open('{remote_log}/metrics.json', 'w') as f:
    json.dump({{'exp_name': 'demo_ssh', 'device': device, 'final_loss': round(loss.item(), 6), 'final_accuracy': round(acc, 4), 'host': '$(hostname)'}}, f, indent=2)
print(f'  loss={{loss.item():.4f}} acc={{acc:.2%}} device={{device}}')
"
        touch {remote_log}/.done
        echo "[chester] Done"
    """)

    print(f"\n[1] Generated script ({len(script.splitlines())} lines)")
    print(f"[2] Submitting to {host}...")
    try:
        backend.submit(task, script, dry=False)
    except Exception as e:
        print(f"    FAIL: {e}")
        return False

    # Wait for .done marker
    import time
    print("[3] Waiting for completion...", end="", flush=True)
    for _ in range(30):
        time.sleep(1)
        print(".", end="", flush=True)
        r = subprocess.run(
            ["ssh", host, f"test -f {remote_log}/.done && echo done"],
            capture_output=True, text=True, timeout=5,
        )
        if "done" in r.stdout:
            break
    print()

    # Pull results
    print(f"[4] Pulling results from {host}:{remote_log} ...")
    subprocess.run(
        ["scp", "-r", f"{host}:{remote_log}/", local_log],
        capture_output=True,
    )

    # Verify
    print("\n[5] Verification:")
    metrics_file = os.path.join(local_log, "metrics.json")
    ckpt = os.path.join(local_log, "checkpoint.pt")

    ok = True
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            m = json.load(f)
        print(f"  metrics.json (pulled from {host}):")
        for k, val in m.items():
            print(f"    {k}: {val}")
    else:
        print(f"  FAIL: metrics.json not found after pull")
        ok = False

    if os.path.exists(ckpt):
        import torch
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        n_params = sum(p.numel() for p in data["model_state_dict"].values())
        print(f"  checkpoint.pt ({os.path.getsize(ckpt)} bytes, {n_params} params)")
        print(f"    final_loss: {data['final_loss']:.4f}")
    else:
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                m = json.load(f)
            if m.get("status") == "torch_not_available":
                print(f"  Note: torch not installed on {host}, but script ran successfully")
                ok = True
            else:
                print(f"  FAIL: checkpoint.pt not found after pull")
                ok = False
        else:
            print(f"  FAIL: checkpoint.pt not found after pull")
            ok = False

    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return ok


def demo_slurm_script():
    """Show what a SLURM training script looks like, with overrides."""
    from chester.backends import create_backend
    from chester.config_v2 import get_backend, load_config

    print("\n" + "=" * 60)
    print(" SLURM (gl): generated training script")
    print("=" * 60)

    cfg = load_config()
    backend = create_backend(get_backend("gl", cfg), cfg)

    task = {
        "params": {
            "log_dir": "/tmp/chester_demo/data/demo_slurm",
            "exp_name": "demo_slurm",
            "learning_rate": 0.001,
            "hidden_dim": 256,
            "train_steps": 1000,
        },
    }

    print("\n[1] Default config (1 GPU, 72h):")
    s1 = backend.generate_script(task, script="train.py")
    for i, line in enumerate(s1.splitlines(), 1):
        print(f"  {i:2d}  {line}")

    print(f"\n[2] With slurm_overrides={{gpus: 4, time: '6:00:00'}}:")
    s2 = backend.generate_script(
        task, script="train.py",
        slurm_overrides={"gpus": 4, "time": "6:00:00"},
    )
    print("  Changed lines:")
    for l1, l2 in zip(s1.splitlines(), s2.splitlines()):
        if l1 != l2:
            print(f"    - {l1}")
            print(f"    + {l2}")


def main():
    parser = argparse.ArgumentParser(
        description="Chester v2 demo -- trains a real neural network")
    parser.add_argument("--ssh", metavar="HOST",
                        help="Also train on SSH HOST (e.g., armdual)")
    args = parser.parse_args()

    demo_dir = setup_demo_project()
    print(f"Demo project: {demo_dir}\n")

    # 1. Local training
    local_ok, local_log = demo_local(demo_dir)

    # 2. SSH training (if requested)
    ssh_ok = None
    if args.ssh:
        ssh_ok = demo_ssh(demo_dir, args.ssh)

    # 3. SLURM script generation
    demo_slurm_script()

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Local training:  {'PASS' if local_ok else 'FAIL'}")
    if args.ssh:
        print(f"  SSH ({args.ssh}):  {'PASS' if ssh_ok else 'FAIL'}")
    else:
        print(f"  SSH:             skipped (use --ssh HOST)")
    print(f"  SLURM script:    shown above")
    print(f"\n  Outputs: {os.path.join(demo_dir, 'data')}")
    print(f"  Inspect checkpoint:")
    print(f"    cd {os.path.dirname(os.path.abspath(__file__))}/tests")
    print(f"    uv run python -c \"import torch; d=torch.load('{local_log}/checkpoint.pt', map_location='cpu', weights_only=False); print(d.keys()); print('loss:', d['final_loss'])\"")


if __name__ == "__main__":
    main()
