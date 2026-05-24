# GPT-OSS-120B Recipes

Production-ready deployment for **GPT-OSS-120B** using TensorRT-LLM on Blackwell (GB200) hardware.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg**](trtllm/agg/) | 4x GB200 | Aggregated | ARM64, MoE communication forced to ALLGATHER |
| [**trtllm/disagg**](trtllm/disagg/) | 5x Blackwell (GB200/B200) | Disaggregated | Prefill/Decode split |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with GB200 (Blackwell) GPUs
3. **HuggingFace token** with access to the model

## Quick Start

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model (update storageClassName in model-cache/model-cache.yaml first!)
kubectl apply -f model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/gpt-oss-agg-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Expected Performance

The `trtllm/agg` benchmark defaults to 512 concurrent requests per GPU on 4x
GB200, for 2048 total concurrent streaming chat requests with ISL=128 and
OSL=1000. On the TensorRT-LLM 1.2.0 release-candidate runtime, five benchmark
runs averaged:

| Metric | Value |
|--------|-------|
| Output throughput | ~105,000 tokens/sec |
| Request throughput | ~106 requests/sec |
| TTFT avg / p99 | ~1.77s / ~2.63s |
| ITL avg / p99 | ~17.3ms / ~19.0ms |
| Request latency avg / p99 | ~18.8s / ~20.4s |
| Tokens/user/sec | ~58 |

## MoE Communication Backend

On GB200 (SM100), the aggregated TensorRT-LLM recipe sets
`TRTLLM_FORCE_COMM_METHOD=ALLGATHER` on the worker container to bypass an upstream
TRT-LLM / DeepEP issue tracked by DYN-2761. Without this override, TRT-LLM can
auto-select DeepEP for GPT-OSS-120B with attention DP and EP=4, which fails
during executor warmup on GB200.

After the upstream DeepEP path is fixed, remove `TRTLLM_FORCE_COMM_METHOD` or set
it to `DEEPEP` to re-enable TRT-LLM's automatic MoE communication selection.

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying
- This recipe requires ARM64 (GB200) nodes — it will not run on x86 Hopper/Ampere hardware
- Update the container image tag in `deploy.yaml` to match your Dynamo release version
