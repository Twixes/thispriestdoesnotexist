# Inactive Cloudflare candidate review

2026-09-21. Reviewed `inference/cloudflare/`, the Python inference server, shared model validator, root configuration, VPS workflow, and the installed `@cloudflare/containers` 0.3.7 implementation. No deployment, authentication change, paid upgrade, or vault access. No implementation edits were needed.

## Findings

No concrete blocking security, deployment-gate, or runtime bug found in this review. The candidate remains inactive: root `wrangler.jsonc` serves static assets, the generated candidate config is ignored, and the candidate CI script is not the current deploy command.

- **Exact weights and review:** the CI validator, generated-config step, Docker verification stage, and Python startup all check the actual safetensors digest against the manifest and its explicit review approval. Positive training progress is required. The Worker checks the loaded model's reviewed status and expected digest before generation, then checks the generation response's digest again. An old image during rollout fails closed. Approval is a repository review record, not an automated claim of visual quality; none of these checks justify approving the rejected pilot.
- **CI identity:** the optional script requires Workers Builds, `master`, and equality between the reported CI commit and checkout HEAD. It validates and prebuilds the image before Wrangler runs. These guards prevent accidental local/non-production use; CI environment variables are not an independent security boundary against a person who already controls execution. The existing VPS workflow also triggers on `master`; its documented disablement is still required if Containers is selected, to avoid two deployment targets reacting to a promoted bundle.
- **Private routing:** only GET `/generate` and `/health` enter the container path. A constant Durable Object name prevents request paths or nonces from creating instances. Caller headers, cookies, query values, and authorization are discarded. The Worker does not expose arbitrary container URLs, ports, RPC methods, or a public origin. Only expected response metadata is returned. Public `/health` deliberately exposes a sanitized model status and can wake the container; it is not a private admin endpoint.
- **Runtime binding:** the configured Durable Object class/export/binding names agree. The pinned SDK supports the used `containerFetch` overload, propagates the abort signal through startup and fetch, and uses the configured default port. The 45-second deadline, bounded Python admission, uncached 503 handling, disabled outbound internet, and non-root image match the intended service.

## Existing checks run

- `npm test --prefix inference/cloudflare`: 6 passed.
- `python3 -m unittest discover -s infra -p 'test_*.py' -v`: 6 passed.
- `research/.venv/bin/python -m unittest inference.test_server -v`: 6 passed, including real tiny-generator forwards and HTTP overload recovery.
- `npm run check` and root `node --test`: passed; root suite reports 22 tests, including the same 6 candidate tests.
- `bash -n inference/cloudflare/ci-deploy.sh`: passed.

## Limits before activation

This was code review and existing local tests, not a Cloudflare integration test. No approved production bundle exists, so the dedicated production-model image and generated config have not been built against one. Account plan/custom-instance eligibility, CI token permissions, real cold starts, rollout behavior, cost, and post-deploy model identity still need verification before activation. The documented two-step frontend promotion remains necessary because Worker activation and container rollout are not atomic. Public health polling can extend billable uptime, and `max_instances: 1` does not enforce a monthly spending cap.

Archive/push readiness is separate from deployment readiness. The candidate can be retained as inactive research infrastructure; it must not advance the quality approval of any model.
