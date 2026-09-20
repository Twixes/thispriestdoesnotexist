# Personal CPU inference origin

This directory prepares the server and the GitOps release path. **No inference server has been purchased, configured, or deployed yet.** The production website stays on its existing Cloudflare deployment until a trained model passes review and the origin is verified.

## Hosting decision, checked 2026-09-20

Use a dedicated personal Hetzner Cloud project in the `michal@matloka.com` account. The existing saved personal login was verified using its email verification code. No work account or work credentials were used. A names-only personal 1Password inventory found only the Hetzner login, with no API-key field or separate Hetzner API item; no local hcloud CLI/config or Hetzner environment-variable names were present. Native browser inspection became unavailable immediately after sign-in, so project ownership, actual stock, billing tax, and the final checkout total still require console verification.

The current [Hetzner price notice](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/) lists Germany/Finland CX23 at $6.49/month or €5.49/month and CX33 at $9.99/month or €8.49/month, excluding tax and IPv4. Prefer CX33 (4 vCPU, 8 GB RAM, 80 GB NVMe) if the final total remains below $15/month; otherwise CX23 (2 vCPU, 4 GB RAM, 40 GB NVMe). [The product page](https://www.hetzner.com/cloud/cost-optimized/) says availability is limited and currently marks these plans unavailable; the console is authoritative when ordering. Do not silently substitute a more expensive CPX plan. [Primary IPv4 costs €0.50/month excluding VAT](https://docs.hetzner.com/cloud/servers/overview/); it simplifies GitHub and package access compared with IPv6-only hosting. No backups, volumes, or other paid add-ons are needed for this reproducible service.

Cloudflare Containers would simplify integration, but [its metered memory and CPU pricing](https://developers.cloudflare.com/containers/platform/pricing/) does not provide a hard $15 monthly cap for an always-ready inference service. Retain the existing Cloudflare frontend and use the fixed-price origin. Measure the actual CPU service before accepting performance; shared vCPUs are not guaranteed dedicated cores.

## Release contract

- Production branch: `master`, direct pushes, no PR requirement.
- Frontend: existing connected Cloudflare Workers Builds.
- Inference: `.github/workflows/deploy-inference.yml`, triggered by relevant changes on `master`.
- Bundle: `models/production/model.json`, `generator.safetensors`, and the export's `LICENSE.txt`. Weights stay in Git LFS; deployment fetches only this directory.
- Runtime: `inference/Dockerfile`, port 8080; `GET /health` exposes model identity and readiness; `GET /generate` returns one newly synthesized WebP with no-store headers.
- Public TLS origin hostname: `inference.thispriestdoesnotexist.com` (proposed, not yet created). Cloudflare's site Worker proxies `/generate` using its private bearer token. Browser JavaScript never gets that token.
- One inference operation at a time, two CPU threads, bounded admission. Overload returns HTTP 503 with Retry-After rather than a cached portrait.

The CI workflow intentionally skips deployment until the production manifest exists. Once present, it fails unless the model is trained, the weights match the manifest, and visual approval is bound to that exact SHA-256. Runtime independently checks the manifest and weights. A manifest alone is not evidence of visual quality; review must first inspect fresh outputs for quality, identity diversity, adult male priest appearance, visible collars, and no hats.

## Provisioning and secrets

1. Review the concrete personal-account order with its tax-inclusive monthly total. Create only the approved server, using Ubuntu 24.04 and the owner's existing personal admin SSH key. No application deployment occurs during provisioning.
2. Create a dedicated CI SSH key. Keep its private half in the personal credential store and deliver it to the repository secret `INFERENCE_SSH_KEY`; never commit or print it. Copy only its public half to the new server.
3. Run `bootstrap-origin.sh` on that fresh server, supplying `ORIGIN_HOSTNAME` and `DEPLOY_PUBLIC_KEY_FILE`. It installs Docker, Caddy, Git LFS, and a restricted `priest-deploy` account. Its SSH key has a forced command, no forwarding, and sudo access only to the validated deployment entrypoint. Existing admin SSH access remains unchanged.
4. Independently verify the origin's SSH host key through the provider console/admin session. Set repository variables `INFERENCE_HOST` (IP/hostname) and `INFERENCE_HOST_KEY` (key type plus public base64 key). CI uses strict host-key checking; do not trust an unauthenticated ssh-keyscan result as verification.
5. Privately provision root-only `/etc/priest/runtime.env` with `INFERENCE_TOKEN=<random value>` and the same token as a Cloudflare Worker secret. The token's source of truth stays in personal credentials. This is application authentication, not a Cloudflare or Hetzner account-wide token.
6. Point the chosen origin DNS hostname at the server so Caddy can issue and renew its public TLS certificate. Allow TCP 80/443 for TLS and origin traffic; restrict administrative SSH to the intended paths. The model process binds only on loopback, behind Caddy.
7. Promote a reviewed model through a `master` commit and push. GitHub CI calls the forced SSH command with the exact commit SHA. The server refuses arbitrary commands and skips superseded revisions. It builds the corresponding image, starts an alternate loopback slot, verifies readiness and model checksum, then reloads Caddy before stopping the old process. No manual local deploy commands.
8. Verify external TLS, auth, fresh seed/body per request, model identity, overload response, and representative image quality. Only then switch the frontend to the server generator in a further CI-deployed commit.

`deploy-origin.sh` is installed root-owned by bootstrap. If its implementation changes, update this infrastructure entrypoint through an authenticated admin setup step; model and application release still run exclusively from CI. This avoids granting the deployment account a general root shell.

## Verification performed locally

`python3 -m unittest discover -s infra -p 'test_*.py' -v` checks valid promotion, changed-weight rejection, review mismatch rejection, untrained/unreviewed model rejection, and forced SSH command restrictions. `bash -n infra/*.sh` checks shell syntax. These tests do **not** prove that an unprovisioned cloud origin works; actual Docker startup, TLS and deployment switchover still need a provisioned server.
