# Security Policy

## Reporting a vulnerability

We take security issues in Switchyard seriously. **Please do not file a public GitHub issue for security vulnerabilities.**

To report a vulnerability, please email NVIDIA's Product Security Incident Response Team (PSIRT) at **psirt@nvidia.com**. NVIDIA PSIRT will acknowledge your report and coordinate any required fix and disclosure timeline.

For details on NVIDIA's product security process, see <https://www.nvidia.com/en-us/security/>.

When reporting, please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, ideally with a minimal example.
- The version of Switchyard affected (commit SHA or release tag).
- Any known mitigations or workarounds.
- Whether you would like to be credited in the public disclosure.

## Supported versions

Switchyard follows a rolling-release model on the `main` branch. Security fixes are landed on `main` and included in the next published release. Older release tags are not patched separately unless the issue is severe and the older version is still in broad use.

| Version | Security fixes |
| --- | --- |
| Latest `main` and most recent release | Yes |
| Older releases | Best-effort only |

## Disclosure timeline

NVIDIA PSIRT typically follows a coordinated-disclosure process:

1. **Acknowledgement** — within 5 business days of report.
2. **Triage and reproduction** — within 30 days.
3. **Fix development and validation** — timeline depends on severity and complexity.
4. **Public disclosure** — coordinated with the reporter; CVEs are filed where appropriate.

We ask that reporters refrain from public disclosure until a fix is available and a disclosure date has been agreed.

## Scope

In scope for security reports:

- Vulnerabilities in the Switchyard library or CLI that allow privilege escalation, credential leakage, denial of service against a Switchyard server, request smuggling, or unauthenticated access to protected endpoints.
- Supply-chain issues in Switchyard's published artifacts.

Out of scope:

- Vulnerabilities in upstream LLM providers (OpenAI, Anthropic, etc.) — report to those providers directly.
- Findings against your own private fork of Switchyard that are not present in `main`.
- Theoretical issues without a demonstrated impact.
