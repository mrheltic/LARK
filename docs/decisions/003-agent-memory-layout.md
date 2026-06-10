# ADR 003: Layered agent memory layout

**Status:** Accepted  
**Date:** 2026-06  
**Context:** AI-assisted development across multiple models/tools

## Problem

Project knowledge was implicit in chat history, empty `CLAUDE.md`, and a long
operations README. Different AI tools (Cursor, Claude Code, others) use different
default context files.

## Decision

Adopt a **layered, model-agnostic** documentation stack:

| Layer | File | Role |
|-------|------|------|
| Entry | `AGENTS.md` | Short manifest for any agent |
| Always-on (Cursor) | `CLAUDE.md` | Minimal summary + pointer to AGENTS.md |
| Deep dive | `docs/architecture.md`, `docs/doa-pipeline.md` | Stable technical reference |
| Decisions | `docs/decisions/*.md` | ADRs for non-obvious choices |
| Operations | `apps/doa_iridium/README.md` | Commands, field procedures |
| Scoped rules | `.cursor/rules/*.mdc` | Cursor-specific always/glob rules |

Do **not** store ephemeral tuning results or session-specific state in permanent docs.

## Rationale

- Single 500-line context file exceeds useful token budget and goes stale.
- ADRs capture **why** without bloating the entry point.
- Cursor rules activate domain context only when editing relevant paths.

## Consequences

- Agents should read `AGENTS.md` first, then follow links.
- Human presentation material can be derived from `docs/doa-pipeline.md`.
- Update ADRs when making architectural decisions, not after every experiment.

## References

- `AGENTS.md`
- `.cursor/rules/project-core.mdc`
