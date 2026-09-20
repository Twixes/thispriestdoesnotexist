# Cloudflare Containers feasibility

Read-only investigation, 2026-09-20. No plan upgrade, new authorization, container publication, deployment, or paid resource was created. The fixed-price VPS scripts remain the prepared default; this document describes an alternative, not an active configuration.

## Account and access

Existing Wrangler 4.135.0 OAuth authentication confirms the personal `michal@matloka.com` account, `1a5e44710a69d21d6822a460e5861578`. `wrangler whoami --account ... --json` reports account type `standard`; this is **not evidence of the Workers Paid subscription**. The current credential has account read and Workers permissions but no `containers:write` scope. `wrangler containers list` refuses locally with that missing-scope error. We did not reauthenticate, read credentials, or invoke 1Password.

Containers require Workers Paid. The account's exact current subscription and Containers entitlement could not be verified with the available CLI authorization. Before adopting this option, verify the personal account's billing plan and enable only the required personal-account Containers/Workers permissions for its CI credential. Existing permission to deploy the static Worker does not establish permission to publish container images. [Containers overview](https://developers.cloudflare.com/containers/)

## Runtime fit

The existing Linux amd64 image serves port 8080 and performs CPU inference without runtime file writes. Its inspected image size is 362,030,838 bytes before adding the approximately 120 MB model bundle. An earlier local Linux amd64 emulation test used about 498 MiB steady memory under a 3 GiB limit. These are local measurements, not Cloudflare measurements.

Candidate instance types:

| Type | vCPU | Memory | Disk | Assessment |
| --- | ---: | ---: | ---: | --- |
| `lite` | 1/16 | 256 MiB | 2 GB | Insufficient memory for the measured service |
| `basic` | 1/4 | 1 GiB | 4 GB | Local test passed, with modest memory headroom and slow cold start |
| custom | 1 | 3 GiB | 4 GB | More CPU headroom, higher idle bill |

Custom instances require at least one vCPU and at least 3 GiB per vCPU. Do not assume a custom one-vCPU/one-GiB instance is supported. Container image size must fit the instance disk. [Instance constraints](https://developers.cloudflare.com/containers/platform/limits/)

A bounded local test at `basic` limits completed three distinct forwards. Cgroup peak memory was 894.64 MiB, Python RSS peaked at 595.46 MiB, process start to first image took approximately 29.89 seconds, and later forwards took 3.50/4.62 seconds. Full evidence and reproducible command: [basic simulation](../research/runs/inference-cpu/cloudflare-basic-simulation/README.md). This demonstrates that this 256-pixel model fits the cap locally, with approximately 129 MiB headroom; it does not establish Cloudflare speed or guarantee that a future larger model fits. It also shows why costs must include repeated Python/model initialization, not just warm inference.

The follow-up [one-CPU simulation](../research/runs/inference-cpu/cloudflare-one-cpu-simulation/README.md) used 3 GiB memory and one inference thread. It completed four unique images, with approximately 5.13 seconds from Python start to the first image, 0.71 seconds median across three warm forwards, and 890.61 MiB cgroup peak memory. Prefer **custom one vCPU / 3 GiB / 4 GB disk, one instance, 30-second idle sleep** for a low-traffic trial. It substantially improves measured latency while the cost scenarios below remain within the preferred budget at modest use. Actual Cloudflare startup and billing still require measurement; no paid trial has been started.

Bake only the reviewed production bundle into a dedicated Docker image, preserving its license. The current VPS Dockerfile expects a mounted model and cannot be used unchanged. Cloudflare's disk resets on sleep; relying on a persistent local model download is unsuitable. Cold starts include Python/PyTorch initialization and model loading, so the platform's typical 1–3 second startup statement is not an application latency guarantee. [Container lifecycle and FAQ](https://developers.cloudflare.com/containers/faq/)

## Cost model

Workers Paid starts at $5/month. Containers add the following usage after the shared monthly allowances:

| Resource | Included | Overage |
| --- | ---: | ---: |
| Provisioned memory | 25 GiB-hours | $0.009/GiB-hour |
| Actual CPU time | 6.25 vCPU-hours | $0.072/vCPU-hour |
| Provisioned disk | 200 GB-hours | $0.000252/GB-hour |

Memory and disk accrue while awake, even when idle; CPU follows active use. Sleeping stops these charges. North America/Europe egress includes 1 TB/month, then $0.025/GB. Workers, Durable Objects, logs, taxes, and other account usage must also be considered. [Official Containers pricing](https://developers.cloudflare.com/containers/platform/pricing/)

For one instance with 4 GB disk, our estimate in USD before tax is:

```text
5 + max(0, memory_GiB * awake_hours - 25) * 0.009
  + max(0, cpu_hours - 6.25) * 0.072
  + max(0, 4 * awake_hours - 200) * 0.000252
```

These are resource scenarios, not promised request counts. Each cold start consumes CPU and awake time in addition to image generation; a burst of requests sharing one start costs less than the same requests spread apart.

| Monthly awake time | Actual CPU time | `basic` | Custom 1 CPU / 3 GiB |
| ---: | ---: | ---: | ---: |
| 10 hours | 1 hour | $5.00 | $5.05 |
| 100 hours | 10 hours | $6.00 | $7.80 |
| 250 hours | 25 hours | $8.58 | $13.08 |
| 720 hours | 100 hours | $18.68 | $31.64 |

At continuous full quarter-CPU utilization for a 30-day month, `basic` reaches approximately $24.44 before tax. A one-instance limit therefore does not guarantee a $15 bill. Low traffic with short idle shutdown is plausibly within budget; a fixed-price VPS offers the stronger cost guarantee. Keep tax headroom and use an application-level admission quota if a strict budget is required; the quota would intentionally reject excess generations.

Workers Paid includes 10 million requests and 30 million CPU milliseconds. Durable Objects includes one million requests and 400,000 GB-seconds; one continuously active 128 MB object for a 30-day month uses about 331,776 GB-seconds. A small, single-instance service should fit those allowances if other personal-account services have not consumed them. No additional $5 DO subscription is needed. [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/), [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/)

## CI-only implementation path

1. Finish model training and visual review. Preserve the exact-hash approval gate; the current pilot is not production approved.
2. Add a dedicated image that copies `models/production` after `infra/validate-model.py` passes. Pull only `public/**,models/production/**` from Git LFS in CI. Run the existing checks and inference tests before deployment.
3. Add a `Container` subclass using port 8080, `CPU_THREADS=1`, `sleepAfter="30s"`, and one stable instance ID, with `max_instances: 1`. Use the default idle-stop behavior. Keep internet access disabled if the image is fully self-contained. All requests share that instance; generation still draws a new random latent every time. Existing bounded queue and 503 responses remain important. [Container class reference](https://developers.cloudflare.com/containers/reference/container-class/)
4. Keep deployment on pushes to `master`. Workers Builds supports Dockerfile builds and its full `wrangler deploy` command publishes the image and rolls out instances. The existing connected build can therefore serve this route without a separate VPS or SSH secret. `wrangler versions upload` does not publish/roll out containers. [CI deployment documentation](https://developers.cloudflare.com/containers/guides/deploy/)
5. First deploy the backend route without switching the gallery. Check `/health` against the reviewed model hash and make real fresh-image requests after both cold and warm starts. Confirm billed usage and latency before switching the frontend in a second CI release.

Wrangler activates Worker code before the image build/push/rollout completes; this is not transactional. Keep old/new Worker and container interfaces compatible, retain the previous image, and verify the actual container after CI succeeds. First provisioning can take minutes. A successful Worker build alone is not proof that inference is ready. No active Wrangler settings, CI configuration, or production routes were changed for this investigation. [Deployment ordering](https://developers.cloudflare.com/containers/guides/deploy/)

## Supported authorization update (prepared, not executed)

Wrangler's documented way to change OAuth scopes is to run `login` again with an explicit whitespace-separated scope list. Merely refreshing an expired access token does not grant missing permissions. Preserve the current scope set and add Containers:

```sh
npx wrangler login --use-keyring --scopes \
  account:read user:read workers:write workers_scripts:write \
  workers_routes:write zone:read containers:write
```

This opens a Cloudflare consent flow; it does not need a 1Password read if the personal browser session is already authenticated. Confirm the personal account in that flow. `--browser=false` prints the consent URL instead, and `--device` supports a code-based flow if the callback is inaccessible. Do not run bare `login` for this task because its default requests every available scope. Only `login --help` and `login --scopes-list` were run during research. [Official login options](https://developers.cloudflare.com/workers/wrangler/commands/general/#login)

Installed Wrangler 4.135.0 confirms `containers:write` is supported and automatically adds `offline_access`; it need not be included manually. Its Containers list, registry image commands and deployment setup use the Containers scope. There is no evidence that this image path requires the unrelated `artifacts:write` or `cloudchamber:write` scopes.

Local OAuth and Workers Builds credentials are separate. For the existing CI build token, retain the permissions that already deploy this Worker and add **Account → Containers → Edit** (the equivalent permission is named **Containers Write** in the newer API vocabulary), scoped only to account `1a5e44710a69d21d6822a460e5861578`. The necessary deployment capabilities are:

| Scope | Permission | Purpose |
| --- | --- | --- |
| Personal account | Containers Edit / Write | Container applications and managed image registry |
| Personal account | Workers Scripts Edit / Write | Worker code, assets and Durable Object bindings/migrations |
| Personal account | Account Settings Read | Wrangler account lookup |
| This site's zone | Workers Routes Edit / Write; Zone Read | Existing custom domain / route reconciliation |
| Personal user | User Details Read; Memberships Read | Existing user-token account discovery |

Workers Builds currently supports user-owned build tokens, not account-owned tokens. Its default generated token includes Workers Scripts, Account Settings, KV, R2, Routes and user discovery, but **does not list Containers**. Editing the existing build token's permissions avoids another secret copy; verify the build still selects that token afterward. KV/R2 permissions are not newly required by this inference design. [Build token configuration](https://developers.cloudflare.com/workers/ci-cd/builds/configuration/#api-token), [API permission names](https://developers.cloudflare.com/fundamentals/api/reference/permissions/)

The managed Containers registry handles push/pull authentication through Wrangler; no Docker Hub password, R2 key, external registry token, or GitHub secret is needed for that path. `Workers CI Write` is only relevant if changing build triggers through the Builds API, not the container deploy permission being added here. The permission set above is documentation/source-derived; it has not yet been validated by a container deployment with this account. [Managed registry authentication](https://developers.cloudflare.com/containers/guides/image-management/)
