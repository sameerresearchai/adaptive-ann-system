# Adaptive ANN System

## Motivation: Networks That Learn to Adapt

The central theme of this project is **adaptive neural networks** — systems that dynamically evolve and discover their own optimal structure during training.

Rather than designing a fixed network architecture upfront, this system explores how neural networks can:

- **Learn connectivity dynamically**: Discover which connections are essential and which are redundant
- **Evolve structure autonomously**: Allow the network to prune, strengthen, or reconfigure itself based on the learning task
- **Balance efficiency with performance**: Find sparse, efficient architectures without sacrificing accuracy
- **Discover emergent patterns**: Let the network reveal what structure works best for the problem at hand

This mirrors principles found in biological neural systems—where synaptic connections continuously adapt, strengthen, or weaken based on experience and necessity. By making networks adaptive, we move toward models that:
- Self-organize rather than being rigidly designed
- Become more interpretable through emergent structure
- Achieve better resource efficiency
- Potentially develop more robust and generalizable representations

## The Core Concept

Instead of asking "What architecture should I build?", this system asks: "What architecture can the network build for itself?"

The network becomes both the learner and the architect—optimizing not just weights and biases, but its own structural form.

## Paper Library

This repository now includes a scalable `papers/` section for long-form explainers and research notes.

- Library landing page: `papers/index.md`
- DARTS explainer: `papers/darts/index.md`
- Input-Adaptive Hard-Exit DARTS: `papers/input-adaptive-early-exit-darts/index.md`

## Notebooks

Companion experimental notebooks are available under `notebooks/`.

- Input-Adaptive Hard-Exit DARTS notebook: `notebooks/input_adaptive_hard_exit_darts.ipynb`

## GitHub Pages

The repo is configured for GitHub Pages with Jekyll.

### Enable deployment

1. Go to repository **Settings** -> **Pages**.
2. Under **Build and deployment**, select **GitHub Actions**.
3. Push to `main` or `master` to trigger `.github/workflows/pages.yml`.

After deployment, pages are available at:

- Home: `/`
- Paper library: `/papers/`
- DARTS page: `/papers/darts/`
- Input-Adaptive Hard-Exit DARTS page: `/papers/input-adaptive-early-exit-darts/`
