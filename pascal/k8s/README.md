# Kubernetes manifests

The manifests here are deliberately generic. They request a GPU
(`runtimeClassName: nvidia`, `nvidia.com/gpu: 1`) and nothing else, because node
names, storage classes and namespaces describe somebody's cluster rather than
this fork, and do not belong in a public repository.

## Running them

```bash
cp -r pascal/k8s/local.example pascal/k8s/local
$EDITOR pascal/k8s/local/kustomization.yaml     # replace the REPLACE_ME_ values
kubectl create namespace vllm-pascal
kubectl apply -k pascal/k8s/local
```

`pascal/k8s/local/` is gitignored.

## What the overlay has to supply

**Node pinning.** `nvidia.com/gpu` does not distinguish GPU architectures. On a
cluster with more than one, the scheduler will happily place a pod built for
`sm_61` on an Ampere node, where it will fail at load rather than at schedule
time. If every GPU node is Pascal, the pin is unnecessary.

**Storage class**, if the default is unsuitable. The build PVC wants 80Gi that
survives pod restarts: it carries ccache, the build tree, the venv, downloaded
models and the Triton cache. That last one matters more here than on supported
hardware — first startup compiles and autotunes every kernel for `sm_61`, which
takes many minutes, and `TRITON_CACHE_DIR` points into the PVC precisely so a
pod restart does not pay it again.

## Why the build pod sleeps

Compiling vLLM for an unsupported architecture is iterative: each pass finds the
next kernel that will not build for `sm_61`. A pod that sleeps lets each fix cost
an `exec` instead of a fresh pod and another full toolchain install. The GPU is
attached to the same pod so it can run the model after building it, which means
the pod holds the card for as long as it exists.
