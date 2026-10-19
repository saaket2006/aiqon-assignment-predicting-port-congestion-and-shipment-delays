---
trigger: always_on
---

If 'Architecture.md' does not exist, create it when first establishing or analyzing the project's architecture.

Whenever a change materially affects the project's high-level architecture, update `Architecture.md` as part of the same task.

Architecture.md should describe only the high-level system structure, including:
- Major components/services
- Major external dependencies
- Primary data stores
- Major communication/data flows
- Authentication/authorization boundaries
- Important deployment boundaries
- Major AI/LLM/agent/vector/RAG components, when applicable

Do NOT document:
- Individual functions or classes
- Minor implementation details
- UI styling or layout
- Routine bug fixes
- Internal code organization unless architecturally significant
- Every library or dependency

Keep the document concise and technology-aware, but avoid unnecessary implementation details.

When an architectural component is added, removed, replaced, or significantly
changed:
1. Make the code changes.
2. Update `Architecture.md` to reflect the new architecture.
3. Remove or revise information that is no longer accurate.
4. Verify that the final document represents the actual high-level architecture.

Examples of architectural changes include:
- Adding or removing a major service/component
- Changing the primary database or storage architecture
- Introducing or removing a major external service
- Changing authentication/authorization architecture
- Changing major data or communication flows
- Introducing or removing an AI/LLM/RAG/vector component
- Changing deployment or hosting architecture
- Splitting or merging major application boundaries

Before modifying Architecture.md, compare it with the current architecture. If it already accurately represents the architecture, leave it unchanged. If a change has no meaningful architectural impact, do not modify `Architecture.md`.

Never create a detailed technical design document in `Architecture.md`; it is intended to provide a high-level overview of the system.