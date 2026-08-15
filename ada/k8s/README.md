# Kubernetes manifests

Generic by construction: the pod asks for one GPU and mounts a model claim whose
name the overlay supplies. Node names, storage classes, namespaces and claim
names describe somebody's cluster, not this fork.

## Running them

```bash
cp -r ada/k8s/local.example ada/k8s/local
$EDITOR ada/k8s/local/kustomization.yaml     # replace the REPLACE_ME_ values
kubectl create namespace vllm-ada
kubectl apply -k ada/k8s/local
```

`ada/k8s/local/` is gitignored. Base and overlays are siblings because kustomize
refuses to build an overlay nested inside its own base; see `ampere/k8s/README.md`
for the two error messages that arrangement produces.

## It takes the model volume as well as the card

The pod mounts the checkpoint read-only from an existing claim instead of
downloading 16 GiB into a second volume. That claim is almost certainly
`ReadWriteOnce`, so it cannot be held by this pod and the serving deployment at
the same time — and unlike a GPU conflict, this one is worth stating plainly
because the pod will simply sit `Pending` on volume attachment with no obvious
explanation.

Scale the deployment to zero first, and expect the model to be unavailable for
the duration:

```bash
kubectl scale deploy -n <ns> <deployment> --replicas=0
kubectl apply -k ada/k8s/local
# ... measure ...
kubectl delete pod -n <ns> ada-dev && kubectl delete pvc -n <ns> ada-dev
kubectl scale deploy -n <ns> <deployment> --replicas=1
```

Check for live traffic before starting, and confirm the restore reached
`2/2 Running` with a `/health` 200 rather than assuming it.

## What the overlay has to supply

**Node pinning.** `nvidia.com/gpu` does not distinguish architectures. On a
mixed cluster the scheduler will place this pod on whatever has a free GPU, and
an sm_89 conclusion drawn on an Ampere or Pascal card is simply wrong.

**The model claim name**, patched into the `models` volume.

**Storage class** for the work PVC, if the default is unsuitable. It carries the
Triton and vLLM compile caches and the profile traces. The compile cache earns
its keep here: Gemma4 is forced onto Triton attention, so a cold start JITs and
autotunes a kernel per shape, and paying that on every pod restart makes
iterative measurement miserable.
