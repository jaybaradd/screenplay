# AI usage disclosure

This repository was developed with AI coding assistance. The application itself uses AI only behind `backend/providers.py`.

## Runtime AI operations

1. screenplay extraction into Pydantic records;
2. optional continuity-only event repair at the extraction review gate;
3. Google Search-grounded cultural research;
4. cultural-brief normalization without adding claims;
5. six independent adaptation-layer plans;
6. one-scene-at-a-time adaptation;
7. constrained correction patches;
8. character/costume and scene image generation.

Every logical operation records the provider, exact model, stable operation and target metadata, complete local input/output, readable previews, hashes, status, aggregate latency, token usage, estimated cost and Langfuse identifiers in `model_runs`. Every actual provider request—including a malformed response that is retried—is retained in `model_call_attempts`. The exported `ai_usage_log.json` combines both levels even when Langfuse is unavailable.

Langfuse uses one stable trace per human command, one span per graph node/logical AI operation, and one generation per real Gemini request. IDs are metadata rather than trace names. Preview capture preserves readable prompt/output context while redacting credential-shaped fields, query-string credentials and all image bytes.

Gemini usage metadata is normalized into input, output, reasoning, tool-use, cached, total and image-output units. Cost is an explicitly labelled public-list-price estimate based on the versioned local catalogue; it is not represented as the user's actual Google invoice.

## Demonstration mode

`AI_MODE=mock` uses a deterministic parser, cautious no-claim planning output, source-preserving adaptation, and labelled placeholder images. It is suitable for tests and a credential-free engineering demonstration. It must not be represented as a culturally accurate adaptation for any profile.

Every model run records the selected culture ID, immutable profile hash and current cultural-brief revision when available.
Culture profiles provide boundaries and research policy; they do not authorize unsupported factual claims.

## Human responsibility

All extraction, cultural planning, screenplay and visual specifications pass through explicit approval gates. The MVP exposes uncertainty and sources, but user approval is not a substitute for a native-language and regional-cultural reviewer in production.
