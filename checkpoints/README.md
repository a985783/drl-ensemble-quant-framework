# Checkpoints

The public repository keeps only the small stable 4-expert MoE checkpoint set
under `checkpoints/moe/stable/`.

Local experiment outputs, single-run checkpoints, and repeated walk-forward fold
models should not be committed. Regenerate or distribute large model artifacts
through a release asset, object storage bucket, or another explicit artifact
channel.
