"""LLM inference across the pool: llama.cpp RPC splits one GGUF model's layers over several Macs.

The coordinator side mounts under :data:`PREFIX` on the main coordinator (plus ``/v1/*`` at the
root); each contributing Mac runs ``python -m slashcompute.inference.node`` next to the MLX agent.
"""

PREFIX = "/inference"
