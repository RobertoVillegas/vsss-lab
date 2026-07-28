# Contributing

1. Read `AGENTS.md` and the relevant ADRs.
2. Run `just doctor` before changing code.
3. Create a focused branch and keep contracts unchanged unless an ADR is accepted.
4. Run `just lint`, `just build`, and `just test`.
5. Include evidence, benchmark impact when applicable, and rollback instructions.

Do not mix infrastructure, physics, and algorithm changes without an explicit reason.
Use Conventional Commits and signed commits.