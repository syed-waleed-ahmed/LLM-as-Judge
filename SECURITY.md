# Security Policy

## Reporting a vulnerability

If you discover a security issue, please **do not open a public issue**. Instead,
report it privately to the maintainer at **syedwaleedahmed9@gmail.com** with:

- a description of the issue and its impact, and
- steps to reproduce (a minimal proof of concept if possible).

You can expect an acknowledgement within a few days. Please give us a reasonable
window to release a fix before any public disclosure.

## Handling secrets

This project talks to a third-party LLM API and therefore uses an API key:

- The key is read from the environment or a local `.env` file, which is
  **git-ignored** and must never be committed.
- Use `.env.example` as a template; it contains no real values.
- The library never logs secret values.
- If a key is ever exposed, **rotate it immediately** in the
  [Groq console](https://console.groq.com/keys).

## Supported versions

The latest release on `main` receives security fixes.
