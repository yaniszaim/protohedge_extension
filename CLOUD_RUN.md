# ProtoHedge Definitive GPU Run

The cloud runner preserves the locked scientific protocol and changes only the
PyTorch execution device. Production mode requires CUDA and will fail rather
than silently consuming paid VM time on CPU.

## 1. Create the upload bundle locally

```bash
cd /Users/yaniszaim/deephedging
/opt/anaconda3/envs/dh/bin/python make_cloud_bundle.py
```

The resulting archive is `dist/protohedge-cloud.tar.gz`.

## 2. Configure Google Cloud

Run these commands in a local terminal after installing and initializing the
Google Cloud CLI. Replace `YOUR_PROJECT_ID` if needed.

```bash
gcloud init
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com
```

Before creating the VM, confirm that the project has a standard NVIDIA L4 GPU
quota of at least one in `us-central1` and a global GPU quota of at least one.

## 3. Create one on-demand L4 VM

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export ZONE="us-central1-a"
export VM_NAME="protohedge-l4"

gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="g2-standard-8" \
  --provisioning-model="STANDARD" \
  --maintenance-policy="TERMINATE" \
  --restart-on-failure \
  --image-family="ubuntu-2204-lts-amd64" \
  --image-project="ubuntu-os-cloud" \
  --boot-disk-size="100GB" \
  --boot-disk-type="pd-balanced" \
  --no-shielded-secure-boot
```

Do not use Spot for the first run. A preemption in the middle of a task loses
that task's progress, although earlier completed tasks remain safe.

## 4. Install the NVIDIA driver

```bash
gcloud compute ssh "${VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}"
```

On the VM:

```bash
sudo systemctl stop google-cloud-ops-agent || true
curl -L https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
  --output cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver \
  --installation-mode=repo \
  --installation-branch=prod
sudo reboot
```

Reconnect after the reboot and verify the accelerator:

```bash
nvidia-smi
```

The PyTorch wheel supplies its CUDA runtime, so the full CUDA Toolkit is not
required for this project.

## 5. Upload and install ProtoHedge

From the local Mac:

```bash
gcloud compute scp \
  /Users/yaniszaim/deephedging/dist/protohedge-cloud.tar.gz \
  "${VM_NAME}:~/" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --compress
```

On the VM:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv tmux
cd ~
tar -xzf protohedge-cloud.tar.gz
./deephedging/cloud_setup.sh
source ~/deephedging/.venv-cloud/bin/activate
```

## 6. Run the end-to-end GPU smoke test

```bash
cd ~
source ~/deephedging/.venv-cloud/bin/activate
python -u -m deephedging.cloud_empirical_runner \
  --smoke-test \
  --device cuda
```

The first lines must report `CUDA verified`, `selected device=cuda`, and the
NVIDIA L4 accelerator. The command must finish with `passed completion gates`.

## 7. Benchmark one complete paper task

```bash
cd ~
source ~/deephedging/.venv-cloud/bin/activate
nohup python -u -m deephedging.cloud_empirical_runner \
  --device cuda \
  --max-tasks 1 \
  > ~/protohedge-first-task.log 2>&1 &
echo $!
tail -f ~/protohedge-first-task.log
```

The production sweep contains 20 tasks. With roughly 105--110 usable VM hours
after setup and disk costs, the first complete task should take about five
hours or less before launching all remaining tasks.

## 8. Continue the full run

If the first task meets the budget threshold, run:

```bash
cd ~
source ~/deephedging/.venv-cloud/bin/activate
nohup python -u -m deephedging.cloud_empirical_runner \
  --device cuda \
  > ~/protohedge-full-run.log 2>&1 &
echo $!
```

The runner verifies and skips the completed first task. Useful monitoring
commands are:

```bash
tail -f ~/protohedge-full-run.log
nvidia-smi
pgrep -af cloud_empirical_runner
cat ~/deephedging/.deephedging_real_runs/submission_rerun_scientific_v5_gpu/cloud_status/shard_0_of_1.json
```

If SSH disconnects, `nohup` keeps the process running. If the VM reboots, rerun
the same full command; completed ticker-liability tasks are skipped.

## 9. Download the completed results

On the VM:

```bash
cd ~/deephedging/.deephedging_real_runs
tar -czf ~/submission_rerun_scientific_v5_gpu-results.tar.gz \
  submission_rerun_scientific_v5_gpu
```

On the local Mac:

```bash
gcloud compute scp \
  "${VM_NAME}:~/submission_rerun_scientific_v5_gpu-results.tar.gz" \
  /Users/yaniszaim/deephedging/ \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --compress

cd /Users/yaniszaim/deephedging/.deephedging_real_runs
tar -xzf ../submission_rerun_scientific_v5_gpu-results.tar.gz
```

Launch the final notebook with the downloaded run selected and training
disabled:

```bash
cd /Users/yaniszaim/deephedging
PROTOHEDGE_RUN_VERSION=submission_rerun_scientific_v5_gpu \
PROTOHEDGE_RUN_TRAINING=0 \
PROTOHEDGE_RUN_ANALYSIS=1 \
jupyter lab notebooks/protohedge-definitive-empirical-rerun.ipynb
```

## 10. Stop billing

After downloading and verifying the results:

```bash
gcloud compute instances stop "${VM_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"
```

The stopped VM no longer incurs GPU/CPU runtime charges, but its persistent
disk continues to incur a small storage charge until the VM or disk is deleted.
