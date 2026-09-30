# Architecture and reliability notes

## Runtime boundaries

```text
Streamlit UI
    │ HTTP only
    ▼
FastAPI ─────────────── app.sqlite
    │                      revisions, approvals, assets,
    │                      issues, idempotency, model runs
    ▼
Culture profile snapshot + approved cultural brief
    │
    ▼
LangGraph ───────────── checkpoints.sqlite
    │                      one thread_id per project
    ▼
AI provider adapter ─── Gemini or deterministic mock
    │
    └────────────────── optional Langfuse observations
```

The frontend imports no backend repository or model code. All mutation passes through FastAPI. The graph holds revision identifiers and hashes; the full documents live as immutable revisions in `app.sqlite`, while images live on disk behind asset records.

## Durable state machine

The workflow is deliberately closed and deterministic:

```text
validate → extract → continuity → [extraction approval]
→ cultural grounding → six parallel layer plans → synthesis → [plan approval]
→ scene adaptation → preservation verification → [screenplay approval/corrections]
→ visual manifest → [prompt approval]
→ character jobs → [character-image approval]
→ scene-keyframe jobs
```

Each bracketed step is a LangGraph `interrupt()`. Resumption uses the same project ID as `thread_id`. Interrupting nodes have no external side effect before the interrupt, so replay is safe.

## Failure and replay model

- SQLite runs in WAL mode and every repository call commits transactionally.
- Structured model calls are cached by provider, model, operation, prompt version, prompt hash and input hash.
- A crash after a model response but before the next checkpoint therefore reuses the stored response.
- Images are generated to a unique temporary path and atomically renamed only after a valid file is returned.
- One asset ID is one generation unit. A retry increments only that asset's attempt counter.
- Edits create child revisions; approved history is never overwritten.
- Surgical corrections require a target hash and prove all non-target block hashes are unchanged.
- Visual dependencies are invalidated only when a correction changes an action/production dependency.

## Continuity ledger

The model extracts evidence-linked continuity events, not authoritative full snapshots. Application code walks scenes in order
and deterministically applies first observations, acquisitions, transfers, derived props, learned facts, costume changes,
injuries and recoveries. Knowledge carries forward unless an explicit supported event removes it. A continuity-only repair
creates a new extraction revision without rewriting source scenes, characters, dialogue blocks or the StoryContract.

## Cultural profile and safety boundary

Culture-specific boundaries are declarative, versioned profiles rather than Python branches or prompt literals. A project
snapshots its selected profile and hash, then one grounded research result is normalized into a source-linked `CulturalBrief`,
reviewed and frozen. A context compiler supplies only the approved claims, policies and language evidence needed by each node.
Raw research is excluded unless promoted into reviewed, cited profile evidence. Mock mode makes no cultural claims.

Model caches, observations and visual dependencies include project isolation plus the profile/brief identity. A fictional
test profile verifies that production-culture language and visual constraints cannot leak into another profile.

## Production migration

For multi-user production, move application data and LangGraph checkpoints to PostgreSQL/object storage, add authentication and authorization, run workers behind a queue, and add an evaluated native-speaker review process. None of those properties are claimed by this local assignment MVP.
