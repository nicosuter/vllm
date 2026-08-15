# Kubernetes manifests

The manifests here are deliberately generic. They request two GPUs
(`runtimeClassName: nvidia`, `nvidia.com/gpu: 2`) and nothing else, because node
names, storage classes and namespaces describe somebody's cluster rather than
this fork, and do not belong in a public repository.

## Running them

```bash
cp -r ampere/k8s/local.example ampere/k8s/local
$EDITOR ampere/k8s/local/kustomization.yaml     # replace the REPLACE_ME_ values
kubectl create namespace vllm-ampere
kubectl apply -k ampere/k8s/local
```

`ampere/k8s/local/` is gitignored.

## This pod cannot share a box with the production deployment

The dev pod asks for both cards, and on a two-card host that is all of them. A
TP=2 serving deployment on the same node will hold them first, and the dev pod
will sit `Pending` indefinitely rather than failing with anything informative.

Asking for one card instead does not help. Three of the four hypotheses in the
parent README are properties of the TP=2 configuration — the all-reduce cost,
the CUDA-graph mode the engine settles on under spec decode, and the baseline
latency itself. Measuring them on a single card measures a different system.

So the sequence is: scale the serving deployment to zero, apply this, measure,
tear the pod down, scale serving back up. Plan for the model to be unavailable
for the duration; there is no way around it short of a third card.

## What the overlay has to supply

**Node pinning.** `nvidia.com/gpu` does not distinguish GPU architectures. On a
cluster with more than one, the scheduler will place this pod wherever there
are two free GPUs, and sm_86 conclusions drawn on an Ada or Pascal card are
simply wrong. If every GPU node is a 3090 Ti pair, the pin is unnecessary.

**Storage class**, if the default is unsuitable. The PVC wants 120Gi that
survives pod restarts: it carries the checkpoint, the Triton and vLLM compile
caches, and the profile traces. The compile cache matters — a cold
`torch.compile` of this model takes about two minutes per rank, and paying that
on every pod restart makes iterative measurement miserable.

## Why the pod sleeps

The work here is iterative measurement rather than a build: each probe answers
one question and the answer decides the next one. A pod that sleeps lets each
step cost an `exec` instead of a fresh pod, a fresh model load and another cold
compile.
