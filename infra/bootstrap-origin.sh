#!/usr/bin/env bash
# Infrastructure setup only. This installs no application or model release.
# Run on a new personal Ubuntu 24.04 VPS after the exact order is approved.
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Run as root on the new personal VPS' >&2; exit 1; }
: "${ORIGIN_HOSTNAME:?Set the public TLS hostname for this personal origin}"
: "${DEPLOY_PUBLIC_KEY_FILE:?Provide the dedicated CI public key file}"
[[ $ORIGIN_HOSTNAME =~ ^[a-z0-9][a-z0-9.-]+$ ]] || exit 1
grep -Eq '^ssh-ed25519 [A-Za-z0-9+/=]+( .*)?$' "$DEPLOY_PUBLIC_KEY_FILE"
[[ $(wc -l < "$DEPLOY_PUBLIC_KEY_FILE") -eq 1 ]] || exit 1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io git git-lfs curl caddy python3 sudo
systemctl enable --now docker caddy
if ! id priest-deploy >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash priest-deploy
fi
install -d -m 700 -o priest-deploy -g priest-deploy /home/priest-deploy/.ssh
{
  printf 'restrict,command="/usr/local/sbin/priest-ssh-command" '
  cat "$DEPLOY_PUBLIC_KEY_FILE"
} > /home/priest-deploy/.ssh/authorized_keys
chmod 600 /home/priest-deploy/.ssh/authorized_keys
chown priest-deploy:priest-deploy /home/priest-deploy/.ssh/authorized_keys
install -m 755 "$script_dir/ssh-command.sh" /usr/local/sbin/priest-ssh-command
install -m 755 "$script_dir/deploy-origin.sh" /usr/local/sbin/priest-deploy
printf 'priest-deploy ALL=(root) NOPASSWD: /usr/local/sbin/priest-deploy *\n' > /etc/sudoers.d/priest-deploy
chmod 440 /etc/sudoers.d/priest-deploy
visudo -cf /etc/sudoers.d/priest-deploy
install -d -m 700 /etc/priest
printf 'ORIGIN_HOSTNAME=%s\n' "$ORIGIN_HOSTNAME" > /etc/priest/origin.conf
chmod 600 /etc/priest/origin.conf
install -d -m 755 /opt/thispriestdoesnotexist/releases
printf '%s\n' 'Bootstrap ready; install runtime.env privately and configure CI. No model/application deployed.'
