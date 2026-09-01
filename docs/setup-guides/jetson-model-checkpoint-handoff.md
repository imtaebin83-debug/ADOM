# Jetson Semantic20 checkpoint hand-off

This guide transfers the two Git-excluded Semantic20 checkpoints and verifies
the SHA-locked `t4` profiles. Checkpoints, ONNX files, TensorRT engines and run
outputs must remain outside Git.

## Canonical artifacts

| Profile | RunPod source | Jetson destination | SHA256 |
| --- | --- | --- | --- |
| `b0-e0` | `/workspace/adom/runs/semantic20/e0/20260805T122006Z-5c50bfdf2900-b0-full/b0/stage2/best_mIoU_iter_6000.pth` | `~/ADOM/models/checkpoints/b0-e0/best_mIoU_iter_6000.pth` | `d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73` |
| `eadom` | `/workspace/adom/artifacts/eadom-b0-seed42-iter26000/checkpoint.pth` | `~/ADOM/models/checkpoints/eadom/checkpoint.pth` | `f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c` |

Each destination directory must contain exactly one `.pth`. Do not place the
E-ADOM file in the B0-E0 directory or rename one model as the other.

## RunPod to a local computer

On the RunPod source terminal, verify the selected file before sending it. This
example downloads E-ADOM; substitute the B0-E0 source and SHA from the table for
the other profile.

```bash
P=/workspace/adom/artifacts/eadom-b0-seed42-iter26000/checkpoint.pth
sha256sum "$P"
runpodctl send "$P"
```

Keep the sender running and copy its one-time receive code. `runpodctl` is
preinstalled on RunPod. A teammate with authorized RunPod terminal access can
run the same two commands and receive the artifact directly on their machine;
no RunPod API key is required for the one-time transfer itself.

On Linux, macOS or WSL, install `runpodctl` if needed, change to a dedicated
download directory and receive the file.

```bash
command -v runpodctl || wget -qO- cli.runpod.net | sudo bash
mkdir -p "$HOME/adom-checkpoints/eadom"
cd "$HOME/adom-checkpoints/eadom"
runpodctl receive <ONE-TIME-CODE>
sha256sum checkpoint.pth
```

On Windows PowerShell, download the current binary once and receive into a
dedicated directory.

```powershell
New-Item -ItemType Directory -Force "$HOME\adom-checkpoints\eadom"
Set-Location "$HOME\adom-checkpoints\eadom"
Invoke-WebRequest `
  https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe `
  -OutFile runpodctl.exe
.\runpodctl.exe receive <ONE-TIME-CODE>
(Get-FileHash .\checkpoint.pth -Algorithm SHA256).Hash.ToLower()
```

Stop if the received digest differs from the canonical table. A disconnected
transfer is retried by starting `runpodctl send` again and using its new code.

## RunPod or local computer to Jetson

The source can be the RunPod path above or a locally verified download. Start a
new sender and leave it running.

```bash
runpodctl send /absolute/path/to/checkpoint.pth
```

From the Windows PowerShell download directory, use the downloaded executable.

```powershell
.\runpodctl.exe send .\checkpoint.pth
```

On Jetson, install `runpodctl` if necessary and receive into a temporary
directory so an unverified file cannot overwrite the active model.

```bash
command -v runpodctl || wget -qO- cli.runpod.net | sudo bash
R=$(mktemp -d)
cd "$R"
runpodctl receive <ONE-TIME-CODE>
sha256sum checkpoint.pth
```

For E-ADOM, install only after the digest is exactly
`f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c`.

```bash
mkdir -p "$HOME/ADOM/models/checkpoints/eadom"
install -m 0644 checkpoint.pth \
  "$HOME/ADOM/models/checkpoints/eadom/checkpoint.pth"
```

For B0-E0, use its distinct destination and digest.

```bash
mkdir -p "$HOME/ADOM/models/checkpoints/b0-e0"
install -m 0644 best_mIoU_iter_6000.pth \
  "$HOME/ADOM/models/checkpoints/b0-e0/best_mIoU_iter_6000.pth"
```

If the received B0-E0 filename is different, verify its digest before renaming
it to `best_mIoU_iter_6000.pth`. Finally, list both profile directories and
verify every installed digest.

```bash
find "$HOME/ADOM/models/checkpoints" -maxdepth 2 -type f -name '*.pth' -print
sha256sum "$HOME/ADOM/models/checkpoints/eadom/checkpoint.pth"
sha256sum "$HOME/ADOM/models/checkpoints/b0-e0/best_mIoU_iter_6000.pth"
```

## Jetson profile setup

Track the reviewed branch. A Jetson clone restricted to the `jetson` branch must
add the dual-model branch to its remote fetch set first.

```bash
cd "$HOME/ADOM"
B=codex/jetson-eadom-dual-model
git remote set-branches --add origin "$B"
git fetch origin
git switch -c "$B" --track "origin/$B"
```

If the local branch already exists, use `git switch "$B"` followed by
`git pull --ff-only`. Define `t4` in `~/.bashrc` so all arguments reach the
launcher, then reload the shell configuration.

```bash
t4() {
    "$HOME/ADOM/scripts/run_jetson_t4.sh" "$@"
}

source "$HOME/.bashrc"
```

Run exactly one perception profile at a time.

```bash
t4 b0-e0
t4 eadom
```

The launcher ignores a stale external `ADOM_MODEL_CONFIG`, chooses the
profile-owned padding-safe runtime config and checks the checkpoint SHA before
launch. For the two canonical full MMEngine checkpoints it also enables the
PyTorch 2.6+ legacy loader compatibility only after the canonical digest
matches. A custom checkpoint SHA override does not receive that automatic
compatibility behavior.

## Live verification

Run ZED sensors in one terminal and one `t4` profile in another. Duplicate
`/adom_perception` nodes mean `t4` was started more than once and must be reduced
to one process before measuring latency.

```bash
ros2 node list | grep -E 'zed_node|adom_perception'
RGB=/zed/zed_node/rgb/color/rect/image
STAT=/adom/perception/status
MASK=/adom/perception/semantic20_mask
ros2 topic info "$RGB" --verbose
timeout 10 ros2 topic hz "$RGB"
timeout 30 ros2 topic echo "$STAT" --once
```

The RGB topic must have one publisher and one perception subscriber. Successful
inference reports `"state": "ok"`. The repository's Jetson ROS CLI does not
accept `--qos-reliability` on `ros2 topic hz`, so do not add that flag.

The runtime padding contract restores the mask to source-image dimensions. With
the current ZED 640x360 publishing profile, both source and mask must report
height 360 and width 640.

```bash
timeout 10 ros2 topic echo "$RGB" --once --field height
timeout 10 ros2 topic echo "$RGB" --once --field width
timeout 30 ros2 topic echo "$MASK" --once --field height
timeout 30 ros2 topic echo "$MASK" --once --field width
```

On 2026-08-14, target Jetson E-ADOM validation confirmed the frozen SHA, the
padding-safe runtime config, one ZED publisher, one perception subscriber and a
successful status message. The observed status sample reported average FPS
`10.45` and capture-to-perception-output latency `873.35 ms`; this is a single
live observation, not a controlled performance benchmark. Preserve a separate
source/mask dimension record before declaring the padding check complete.
