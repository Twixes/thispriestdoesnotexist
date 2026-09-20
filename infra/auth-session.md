# Authentication without repeated approval prompts

Verified against official documentation on 2026-09-20. Installed CLI: `op 2.39.0`. This project uses the personal `my.1password.com` account, `michal@matloka.com`; never the work account.

## Why the repeated prompts occurred

The desktop integration authorizes one terminal session at a time. New terminal sessions require new authorization. Its idle limit is 10 minutes, refreshed by use, with an absolute 12-hour limit; locking the app revokes authorization. Consequently, running separate credential queries from short-lived execution shells can repeatedly prompt even when the app appears unlocked. These are the documented CLI limits, not a configurable 24-hour authorization setting. [1Password app-integration security](https://www.1password.dev/cli/app-integration-security)

Do not keep the whole vault unlocked, alter global lock settings, poll 1Password just to prolong a session, or repeatedly inspect metadata through authenticated commands. App auto-lock settings affect the app's unlock behavior; they do not provide a documented replacement for the CLI limits. [App lock settings](https://support.1password.com/auto-lock/)

The CLI's `--cache` switch controls an encrypted in-memory performance cache; it does not promise longer authorization. It is already enabled by default on macOS. [CLI cache reference](https://www.1password.dev/cli/reference#cache)

## The supported task-scoped pattern

Fetch the small, explicit set of task credentials **once**, then let the task process retain them. `op run` passes selected secrets to its child as environment variables for that child's lifetime and masks matching stdout/stderr output by default. The child uses the actual service APIs directly afterward; it does not call `op` for each request. This supports a long-running task without periodically reopening 1Password. [Official `op run` reference](https://www.1password.dev/cli/reference/commands/run)

For an interactive automation task, start a single persistent PTY and reuse its process/session handle. If credentials must be loaded from 1Password, invoke `op run` once in that PTY, with an environment-reference file that lists **only** the required items:

```sh
OP_ACCOUNT=my.1password.com op run --env-file=/private/task/path/references.env -- bash --noprofile --norc
```

The reference file contains `op://` references, never secret values. Do not put actual credentials in arguments, shell history, repository files, stdout, or tool messages. Leave masking enabled and shell tracing disabled. Subsequent work goes through the same child process or a narrowly scoped API client that owns the loaded credentials. End that process when the task finishes or at the agreed lifetime, e.g. 24 hours; process-local credentials disappear with it. A service token may have its own shorter expiry.

This is process-scoped credential reuse, not a claim that the 1Password authorization itself lasts 24 hours. If the initial authorization is already expired, **one** initial biometric approval remains necessary to fetch new secrets. It is not possible to recover secret values from previously completed metadata commands: those processes have already exited.

Manual `OP_SESSION` sign-in is not a better fix: its documented idle expiry is 30 minutes and it requires account-level sign-in material. Do not export the master password or Secret Key to automate that route. [Manual CLI sign-in](https://www.1password.dev/cli/sign-in-manually)

## This site's current path

Existing GitHub CLI and Cloudflare/Wrangler authentication already work independently of repeated 1Password reads. The personal Hetzner browser login was completed; reuse that session. The personal vault inventory found a Hetzner login only, with no API token, notes, or extra credential fields. No account-wide vault session or broad credential export is needed.

Once its browser session is accessible, create a token limited to the site's dedicated personal Hetzner project. Hetzner supports read-only tokens for GET requests and read/write tokens for resource administration. Its token is displayed only when created. Keep it in the personal credential store and load it once into the task process; API calls then need no 1Password access. [Hetzner API-token instructions](https://docs.hetzner.com/cloud/api/getting-started/generating-api-token/)

No new authentication prompts, tokens, cache, vaults, or global security-setting changes were created while researching this procedure. A cache cannot yet be populated with a project token because that token does not exist. The immediate operational change is to stop repeated authenticated inventory queries and reuse established service sessions.

## Optional reusable 24-hour vault access

If future tasks need new secrets throughout a long run, 1Password service accounts support an explicit lifetime, such as `--expires-in 24h`, and read access restricted to a dedicated task vault. They cannot access the built-in Private, Personal, Employee, or default Shared vault. Permissions are fixed at creation. This would require a new dedicated automation vault containing only this project's tokens, plus a one-time service-account setup; it is unnecessary for a single already-authenticated hosting workflow. [Service-account setup and restrictions](https://www.1password.dev/service-accounts/get-started)
