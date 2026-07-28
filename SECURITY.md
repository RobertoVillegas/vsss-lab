# Security policy

Report vulnerabilities privately through GitHub Security Advisories. Do not open a
public issue containing secrets or exploit details.

- Never commit credentials, private keys, tokens, checkpoints with secrets, or `.env`.
- Pin OCI images by digest and GitHub Actions by commit.
- Do not mount the Docker socket into training or controller containers.
- Self-hosted GPU runners must not execute untrusted fork code.
- Preserve lockfiles and review automated dependency updates.