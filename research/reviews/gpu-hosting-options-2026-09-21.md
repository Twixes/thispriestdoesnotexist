# GPU hosting feasibility — 2026-09-21

The next useful purchase would be a short, isolated RTX 4090 benchmark, not a monthly server. Verified public offers put one to two hours comfortably within a few dollars. No checked always-resident GPU fits $15/month. Low-traffic serverless can fit that spend, but it does not establish a sub-500ms first request after idle. GPU permission does not resolve either the model quality gate or the latency gate.

This is research only: no resources created, credentials read, account settings changed, purchases made, or production releases performed. Prices are USD before tax; estimates are not bills. Monthly comparisons use 730 hours. Availability and checkout totals must be rechecked when ordering.

## Live marketplace offers

A read-only, unauthenticated POST to [Vast's official search endpoint](https://docs.vast.ai/api-reference/search/search-offers), `https://console.vast.ai/api/v0/bundles`, returned these offers at **2026-09-21 08:58:19 UTC**. Each query selected one on-demand, verified, rentable, currently unrented GPU, at least 50GB available disk, sorted by `dph_total`. Names use spaces (`RTX 4090`), not underscores. These are observed offers, not guaranteed reservations or universal minimum prices.

| GPU / offer ID | Location | VRAM | Compute `dph_base` / hour | Compute / 730h | 50GB storage / month | Compute + 50GB / month |
|---|---|---:|---:|---:|---:|---:|
| RTX 3060 / 48521510 | South Korea | 12GB | $0.049333 | $36.01 | $10.00 | $46.01 |
| RTX 3090 / 34161228 | China, Yunnan | 24GB | $0.146667 | $107.07 | $10.00 | $117.07 |
| RTX 4090 / 34176464 | US, Michigan | ~24GB | $0.386667 | $282.27 | $33.33 | $315.60 |

The corresponding API `dph_total` values were $0.05155556, $0.14888889, and $0.39407407/hour. Those totals reflect the API's default storage allowance, **not an order for 50GB**; the table instead separately calculates 50 × returned `storage_cost` ($0.20, $0.20, $0.66666667/GB/month). Available disk was 127GB, 172GB, and 755.65GB. Returned reliability scores were 0.9973984, 0.9980033, and 0.9889018; these are marketplace fields, not measured uptime for this site. The cheapest 4090 is therefore a benchmark candidate, not a production recommendation.

Vast bills compute by the second. Storage continues while an instance exists, even stopped; bandwidth is additional and host-dependent. For the 4090 offer, reported ingress was $0.00130208/GB and egress $0.00260417/GB. Downloading a 16GB model would therefore add roughly $0.02 at those quoted rates. Marketplace on-demand access avoids the deliberate interruption risk of spot; it does not prove host reliability or latency. Delete a finished benchmark instance to end storage charges. [Vast pricing rules](https://docs.vast.ai/guides/instances/pricing)

A broader query without the verified filter returned a 3060 offer at `dph_total` $0.03555556/hour (offer 20610591, China), approximately $25.96/730h with its default disk allowance. Even that did not reach $15. It is not the recommended reliability baseline. The rendered [4090 pricing page](https://vast.ai/pricing/gpu/RTX-4090) said no offers while the API returned offers; use the live API/checkout, not that stale page.

## Fixed published rates

[Runpod's pricing page](https://www.runpod.io/pricing), marked updated September 13, lists:

| GPU | Pod / hour | Pod / 730h | Serverless / hour |
|---|---:|---:|---:|
| RTX A5000, 24GB | $0.27 | $197.10 | $0.69 grouped tier |
| RTX 3090, 24GB | $0.50 | $365.00 | $0.69 grouped tier |
| RTX 4090, 24GB | $0.74 | $540.20 | $1.10 |
| RTX 5090, 32GB | $0.99 | $722.70 | $1.58 |

These are published rates; the page exposes Community/Secure choices and the actual deployment console determines stock and final tier. No current fixed Runpod 3060 rate was verified. No listed rate proves this pipeline's speed.

Runpod Pod GPU and storage charges accrue per second. Deployment requires credits covering at least one hour, which is distinct from a minimum one-hour rental charge. Container disk is $0.10/GB/month while running; volume disk remains billable when stopped. There is no Pod ingress/egress charge. A 50GB ephemeral disk is approximately $0.00685 per rental hour using the same 730-hour convention. Minimum cash top-up, taxes and account eligibility were not checked. [Pod billing](https://docs.runpod.io/pods/pricing)

## Serverless budget and cold starts

Runpod flex workers scale to zero. Charges include container/model startup, inference, and idle timeout (default five seconds), rounded to whole seconds. Active workers stay running, with discounts available by inquiry. Neither FlashBoot nor model caching establishes a 500ms cold request for this model. [Serverless billing](https://docs.runpod.io/serverless/pricing)

At $1.10/hour, hypothetical **total billed** worker time per portrait gives:

| Billed seconds / delivered portrait | Cost / 1,000 portraits | Portraits for $10 compute |
|---:|---:|---:|
| 10 | $3.06 | 3,272 |
| 30 | $9.17 | 1,090 |
| 60 | $18.33 | 545 |

These are arithmetic scenarios, not measured pipeline performance. Even a 0.5-second generation plus a separate five-second idle period rounds to six billed seconds for an isolated lifecycle: about $1.83/1,000 **before startup**, storage, retries and tax. Closely spaced requests share idle time; sparse requests repeatedly incur startup. Retrying rejected images increases both latency and cost.

[Modal pricing](https://modal.com/pricing) lists T4 at $0.000164/second ($0.5904/hour), L4 at $0.000222/second ($0.7992/hour), A10 at $1.1016/hour, and H100 at $3.9492/hour. CPU and RAM add charges. Starter advertises $30/month included compute; no personal account eligibility or unused credit was verified. This could cover a benchmark with no marginal cash payment, but is not a guaranteed entitlement. A continuously reserved L4 is already $583.42/month GPU-only before the credit. Region selection carries a listed premium.

Modal's `min_containers` holds paid warm capacity. Its default idle window is 60 seconds (configurable from two seconds to 20 minutes); idle GPU reservation is billable. Containers boot in roughly a second before model initialization, which may take seconds to minutes. [Cold-start documentation](https://modal.com/docs/guide/cold-start) GPU memory snapshots are alpha and accelerate restoration; the docs do not promise 500ms end-to-end cold inference. [Snapshot documentation](https://modal.com/docs/guide/memory-snapshots)

## What can meet the two constraints?

The entire $15 budget permits **$0.02055/hour** for 24/7 compute, before storage, tax, domain or frontend costs. Reserving $5 for those leaves $0.01370/hour. None of the verified options approaches this. A resident 3060 is the closest, but its 12GB VRAM and actual generation speed require measurement. Loading/offloading a large text encoder for every request would work against the latency target. Fixed, precomputed prompt embeddings could reduce resident memory and work while retaining fresh stochastic image generation; they do not make a 3060 proven fast enough.

There are two distinct possible contracts: warm generation below 500ms with occasional slow startup, or below 500ms generation plus a consistently responsive first request at any hour. Serverless is plausible financially for the former, subject to benchmark; scale-to-zero does not establish the latter. A pre-generated image queue would not fulfill the user's fresh server generation requirement. An owned, continuously available GPU could avoid rental charges, but availability, electricity and hosting reliability are unknown; no such machine was assumed.

## Concrete benchmark decision

Use a single on-demand **RTX 4090**, bounded to one or two hours, with 50GB ephemeral disk. Prefer a currently verified, reliable Vast host if a personal account is readily available; the observed offer costs about **$0.89 for two hours plus a 16GB download**, before tax. This uses $0.386667 × 2 + ($33.3333 / 730 × 2) + $0.02083. Recheck live host reliability and availability. Runpod is a simpler fixed-rate fallback: approximately **$1.49 for two hours plus 50GB disk**. A **$3 resource budget** gives useful margin without committing to recurring hosting; an account's required cash top-up may exceed resource consumption. Modal's advertised credit is another possible benchmark route, with additional platform integration and eligibility checks.

Benchmark the final candidate and adapter, batch size one, fresh seeds, at intended displayed resolution. Record GPU, driver, precision, compiled/eager setting and exact model/adapter hashes. Measure model load/compilation separately; then run at least 100 sequential warm requests with CUDA synchronization, including diffusion, VAE, any quality checks/retries, and WebP encoding. Report median, p95 and maximum, followed by cold boot, concurrent requests/queueing, and public HTTP latency. Preserve outputs for visual review: speed cannot compensate for rejected faces. Precompute fixed conditioning and compile a resident pipeline if useful, but charge startup time honestly. Export results and terminate the resource after the bounded experiment. This report does not authorize or perform provisioning.

## Existing account/setup evidence

Read-only inspection of `infra/README.md`, `infra/auth-session.md`, `infra/provisioning.md` and `infra/cloudflare-containers.md` found preparation for a personal Hetzner **CPU** origin, with its README explicitly saying no inference server had been bought/configured/deployed. Prior saved personal Hetzner login verification is historical evidence, not a current account check. The frontend is already on personal Cloudflare; the prepared inference release is GitHub Actions from `master`. No Runpod, Vast or Modal setup was documented in these files; that does not mean the user lacks those accounts. No credentials, browser sessions or vault entries were inspected here. Production GPU integration must retain CI-only releases and model review gates.
