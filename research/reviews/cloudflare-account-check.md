# Personal Cloudflare account check

Observed read-only in the existing Arc session on 2026-09-21 (Europe/Warsaw).
Account: Michal@matloka.com's Account, ID `1a5e44710a69d21d6822a460e5861578`, matching the repository configuration.

- The Workers plans page explicitly marks **Free** ($0) as the current plan.
- Paid is offered at **$5/month plus usage**, with Containers included in its feature list. No upgrade was selected or purchased.
- The Containers page displayed no container rows, zero billable usage ($0.00), and zero memory, disk, CPU and egress counters for the August 29–September 29 billing period.
- Seeing the Containers dashboard does not demonstrate that a Free account can deploy containers, nor that the proposed custom 1-vCPU/3-GiB shape is eligible after upgrading. Those remain deployment prerequisites.
- No authentication prompt, vault read, token creation, plan change, container deployment or billing action occurred. Existing personal browser authentication worked.

Observed pages:
- https://dash.cloudflare.com/1a5e44710a69d21d6822a460e5861578/workers/plans
- https://dash.cloudflare.com/1a5e44710a69d21d6822a460e5861578/workers/containers

Only task-relevant observations are recorded; the unrelated browser sidebar and account UI were not archived as screenshots. The inactive candidate continues to require a reviewed model and CI deployment from master.

## Existing CLI authentication

`npx wrangler whoami` (4.135.0) successfully returned the same personal account using its existing encrypted-file/macOS-Keychain OAuth storage. Current scopes are user:read, offline_access, account:read, workers:write, workers_scripts:write, workers_routes:write and zone:read. Containers write access is absent. Wrangler's generic warning lists many unrelated optional scopes; it is not a reason to grant them all. No login refresh or scope expansion was performed.
