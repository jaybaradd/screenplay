# AI usage disclosure

This repository was developed with AI coding assistance. The application itself uses AI only behind `backend/providers.py`.

## Runtime AI operations

1. screenplay extraction into Pydantic records;
2. Google Search-grounded cultural research;
3. cultural-brief normalization without adding claims;
4. six independent adaptation-layer plans;
5. one-scene-at-a-time adaptation;
6. constrained correction patches;
7. character/costume and scene image generation.

Every runtime call records the provider, model, operation, prompt and input hashes, status, latency and any error in `model_runs`. The exported `ai_usage_log.json` is produced from those local records even when Langfuse is unavailable. Full screenplay input/output is omitted from Langfuse by default.

## Demonstration mode

`AI_MODE=mock` uses a deterministic parser, cautious no-claim planning output, source-preserving adaptation, and labelled placeholder images. It is suitable for tests and a credential-free engineering demonstration. It must not be represented as a culturally accurate Maidani Mewari adaptation.

## Human responsibility

All extraction, cultural planning, screenplay and visual specifications pass through explicit approval gates. The MVP exposes uncertainty and sources, but user approval is not a substitute for a native-language and regional-cultural reviewer in production.
