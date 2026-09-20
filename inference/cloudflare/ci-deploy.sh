#!/usr/bin/env bash
# OPTIONAL Workers Builds deploy command. Never run this as a local deployment.
set -euo pipefail
[[ ${WORKERS_CI:-} == 1 && ${WORKERS_CI_BRANCH:-} == master ]] || {
  echo 'This deployment command runs only in Workers Builds on master.' >&2; exit 1;
}
[[ ${WORKERS_CI_COMMIT_SHA:-} == "$(git rev-parse HEAD)" ]] || {
  echo 'Build revision does not match checked-out master commit.' >&2; exit 1;
}
git lfs pull --include='public/**,models/production/**' --exclude=''
test -f models/production/model.json
python3 infra/validate-model.py
test -s models/production/LICENSE.txt
npm ci --prefix inference/cloudflare --ignore-scripts
npm test --prefix inference/cloudflare
node inference/cloudflare/prepare-config.mjs
# A local image build fails BEFORE Wrangler activates a Worker on model/image errors.
docker build --platform linux/amd64 -f inference/cloudflare/Dockerfile \
  -t "priest-reviewed:$WORKERS_CI_COMMIT_SHA" .
./node_modules/.bin/wrangler deploy --config inference/cloudflare/wrangler.generated.json
