#!/usr/bin/env bash
set -euo pipefail
revision=${1:?Expected commit SHA}
[[ $revision =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid commit SHA' >&2; exit 1; }
: "${INFERENCE_HOST:?Set the personal origin host repository variable}"
: "${INFERENCE_HOST_KEY:?Set the independently verified SSH host-key repository variable}"
: "${INFERENCE_SSH_KEY:?Set the scoped inference deployment key repository secret}"
[[ $INFERENCE_HOST =~ ^[a-zA-Z0-9.-]+$ ]] || { echo 'Invalid origin hostname' >&2; exit 1; }
key_dir=$(mktemp -d)
trap 'rm -rf "$key_dir"' EXIT
umask 077
printf '%s\n' "$INFERENCE_SSH_KEY" > "$key_dir/key"
printf '%s %s\n' "$INFERENCE_HOST" "$INFERENCE_HOST_KEY" > "$key_dir/known_hosts"
ssh -i "$key_dir/key" -o IdentitiesOnly=yes -o BatchMode=yes \
  -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$key_dir/known_hosts" \
  -o ConnectTimeout=20 "priest-deploy@$INFERENCE_HOST" "deploy $revision"
