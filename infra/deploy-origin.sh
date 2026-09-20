#!/usr/bin/env bash
# Called only by the CI-only, forced SSH command on the dedicated personal VPS.
set -euo pipefail
revision=${1:?Expected commit SHA}
[[ $revision =~ ^[0-9a-f]{40}$ ]] || exit 1
[[ $(id -u) == 0 ]] || { echo 'This command requires root' >&2; exit 1; }
exec 9>/run/priest-deploy.lock
flock -w 1200 9
base=/opt/thispriestdoesnotexist
mkdir -p "$base/releases"
if [[ ! -d $base/repository.git ]]; then
  git clone --bare https://github.com/Twixes/thispriestdoesnotexist.git "$base/repository.git"
fi
git --git-dir="$base/repository.git" fetch origin master
latest=$(git --git-dir="$base/repository.git" rev-parse FETCH_HEAD)
if [[ $latest != "$revision" ]]; then
  echo 'Skipping superseded CI revision; a newer master push will deploy.'
  exit 0
fi
release="$base/releases/$revision"
if [[ ! -d $release/.git ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "$base/repository.git" "$release"
  git -C "$release" remote set-url origin https://github.com/Twixes/thispriestdoesnotexist.git
  GIT_LFS_SKIP_SMUDGE=1 git -C "$release" checkout --detach "$revision"
fi
git -C "$release" lfs pull --include='models/production/**' --exclude=''
expected=$(python3 "$release/infra/validate-model.py" "$release/models/production")
# The token is provisioned separately; it never lives in git or CI build logs.
test -s /etc/priest/runtime.env
grep -q '^INFERENCE_TOKEN=.' /etc/priest/runtime.env
source /etc/priest/origin.conf
[[ ${ORIGIN_HOSTNAME:-} =~ ^[a-z0-9][a-z0-9.-]+$ ]] || exit 1
# The proxy configuration is authoritative after a crash between reload and
# recording state; never kill the currently served slot based on a stale file.
upstream=$(awk '$1 == "reverse_proxy" { print $2; exit }' /etc/caddy/Caddyfile)
case "$upstream" in
  127.0.0.1:8081) active=blue ;;
  127.0.0.1:8082) active=green ;;
  *) active= ;;
esac
if [[ $active == blue ]]; then slot=green; port=8082; else slot=blue; port=8081; fi
candidate="priest-$slot"
docker build -f "$release/inference/Dockerfile" -t "priest:$revision" "$release"
docker rm -f "$candidate" >/dev/null 2>&1 || true
docker run -d --name "$candidate" --restart unless-stopped \
  --memory 3g --cpus 2 --pids-limit 256 --cap-drop ALL \
  --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  --env-file /etc/priest/runtime.env -e CPU_THREADS=2 \
  -p "127.0.0.1:$port:8080" \
  -v "$release/models/production:/app/model:ro" "priest:$revision" >/dev/null
switched=0
cleanup() {
  if [[ $switched == 0 ]]; then docker rm -f "$candidate" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
ready=0
for attempt in $(seq 1 60); do
  if curl --fail --silent --max-time 5 "http://127.0.0.1:$port/health" | \
    python3 -c 'import json,sys; h=json.load(sys.stdin); assert h["status"] == "ready" and h["reviewed"] is True and h["training_step"] > 0 and h["model_sha256"] == sys.argv[1]' "$expected" 2>/dev/null; then
    ready=1; break
  fi
  sleep 2
done
[[ $ready == 1 ]] || { echo 'Candidate failed readiness; current service kept' >&2; exit 1; }
config=$(mktemp /etc/caddy/Caddyfile.XXXXXX)
printf '%s {\n  reverse_proxy 127.0.0.1:%s\n}\n' "$ORIGIN_HOSTNAME" "$port" > "$config"
chmod 644 "$config"
caddy validate --config "$config" --adapter caddyfile
cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.previous
mv "$config" /etc/caddy/Caddyfile
if ! systemctl reload caddy; then
  mv /etc/caddy/Caddyfile.previous /etc/caddy/Caddyfile
  systemctl reload caddy
  exit 1
fi
# Caddy now serves this container. Bookkeeping failures must not remove it.
switched=1
printf '%s\n' "$slot" > "$base/active-slot"
printf '%s\n' "$revision" > "$base/active-revision"
if [[ -n $active ]]; then docker stop --time 120 "priest-$active" >/dev/null; fi
printf 'Deployed master revision %s with reviewed model %s\n' "$revision" "$expected"
