# Runbook: install the SWIM consumers on the spare Mac (Claude runs this over SSH)

Prerequisite: Sean has done `SPARE_MAC_SETUP.md` (Remote Login on, never sleeps, restarts after power
failure) and has given Claude the hostname and short username. Throughout:

```sh
MAC=<shortuser>@<hostname>.local        # e.g. seans@flyrely-server.local
```

Steps marked **[SEAN TTY]** need the Mac's admin password. Claude cannot type it. Sean runs that one
command in his own terminal (the `-t` gives a password prompt); Claude waits. Nothing below prints, logs or
echoes a credential.

**Safety.** The SWIM queues are shared and live. Before the first live start (step 7) make sure no other
consumer (the main Mac, another agent, a smoke test) is bound to the same queues: two consumers on one
non-exclusive queue split the messages between them. Steps 1 to 6 never connect to SWIM.

## 1. Identify the Mac and decide the path

```sh
ssh $MAC 'sw_vers -productVersion; uname -m; df -h / | tail -1; pmset -g | grep -E "^ *(sleep|autorestart|womp)"'
```

Minimum macOS for the SWIM client library (`solace-pubsubplus` on PyPI, wheel tags checked 2026-11-05):

| Wheel | Tag | Minimum macOS | CPU |
|---|---|---|---|
| 1.7.0, 1.8.0, 1.9.0, 1.10.0, **1.11.0 (pinned)**, 1.12.0 | `py36-none-macosx_11_0_universal2` | **macOS 11.0 Big Sur** | Intel and Apple Silicon |
| 1.0.0 to 1.6.0 | `py36-none-macosx_10_15_x86_64` | macOS 10.15 Catalina | Intel only |
| (no macOS wheel, no sdist) | | older than 10.15 | not installable: use the Linux fallback (last section) |

So: **macOS 11 or newer** uses `requirements-swim-minimal.txt` (1.11.0). **macOS 10.15** uses
`requirements-swim-macos10.txt` (1.6.0; its API is older and has not been exercised against the live broker,
so step 6 matters). **Older than 10.15**: stop, use the Oracle VM. `zstandard` 0.25 ships wheels for Python 3.11 on
macOS 10.9+ (Intel) and 11+ (Apple Silicon), so it is never the limiting package; if it fails to install, the
consumer falls back to gzip (archives about 1.8 times bigger).

Disk: about 1 GB per day is expected after filtering (see the PR), 14 days kept, so 20 GB free is enough.

## 2. Python 3.11 from python.org (works back to macOS 10.9)

Apple's system Python is too old or absent, and Homebrew has dropped old macOS versions. The python.org
universal2 installer for 3.11.9 supports macOS 10.9 and later and runs natively on Intel and Apple Silicon.

```sh
ssh $MAC 'cd /tmp && curl -fLO https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg && md5 python-3.11.9-macos11.pkg'
```
Compare the printed MD5 with the one listed on https://www.python.org/downloads/release/python-3119/ (macOS 64-bit
universal2 installer). Do not continue on a mismatch.

```sh
ssh -t $MAC 'sudo installer -pkg /tmp/python-3.11.9-macos11.pkg -target /'     # [SEAN TTY]
ssh $MAC 'rm /tmp/python-3.11.9-macos11.pkg'
ssh $MAC '"/Applications/Python 3.11/Install Certificates.command"'              # lets Python itself reach https (alerts)
ssh $MAC '/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -V'   # expect Python 3.11.9
```

## 3. Copy the code (no git, no Xcode tools needed) and make the venv

From the main Mac, in the repo checkout, after the PR is merged (replace `origin/master` with the branch to test):

```sh
cd /Users/seanbutton/Projects/flyrely/flyrely-api && git fetch -q
ssh $MAC 'mkdir -p ~/flyrely-api'
git archive origin/master | ssh $MAC 'tar -x -C ~/flyrely-api'      # tracked files only: no .env, no data/, no venv
ssh $MAC 'cd ~/flyrely-api && /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv venv && venv/bin/python -m pip install -q --upgrade pip'
```

## 4. Check the solace wheel BEFORE installing (compatibility check)

```sh
ssh $MAC 'cd ~/flyrely-api && venv/bin/python -m pip install --only-binary=:all: --dry-run -r requirements-swim-minimal.txt' | tail -5
```
- Output ends with `Would install ... solace-pubsubplus-1.11.0 ...`: good, continue.
- `No matching distribution found for solace-pubsubplus==1.11.0`: the Mac is older than macOS 11. If it is
  10.15, run the same command with `requirements-swim-macos10.txt`. If that also fails, the Mac is too old: Linux fallback.

Then install for real with the file that passed:
```sh
ssh $MAC 'cd ~/flyrely-api && venv/bin/python -m pip install --only-binary=:all: -r requirements-swim-minimal.txt'   # or ...-macos10.txt
ssh $MAC 'cd ~/flyrely-api && venv/bin/python -c "import solace.messaging, zstandard, certifi, dotenv; print(\"imports ok\")"'
```
(If only `zstandard` fails, remove it from the file; the archive then uses gzip.)

## 5. Credentials: only the SCDS_* lines, by scp

On the main Mac. Only the TFMS and SFDPS lines are copied (STDDS is not deployed yet). The values are never printed.

```sh
cd /Users/seanbutton/Projects/flyrely/flyrely-api
T=$(mktemp -d) && umask 077 && grep -E '^SCDS_(TFMS|SFDPS)_' .env > "$T/swim.env" \
  && scp -q "$T/swim.env" $MAC:flyrely-api/.env && rm -rf "$T"
ssh $MAC 'chmod 600 ~/flyrely-api/.env && grep -c "^SCDS_" ~/flyrely-api/.env'      # prints a COUNT only; expect 10
```
Alert channel (after Sean picks one): put only the `FLYRELY_ALERT_*` variables from `alerts.py` in a separate
`~/flyrely-api/.env.alerts` (mode 600) on the Mac, for example ntfy:
```sh
ssh $MAC 'umask 077; printf "FLYRELY_ALERT_NTFY_TOPIC=%s\n" "<topic>" > ~/flyrely-api/.env.alerts'
```
Test the channel without touching SWIM. It prints the list of channels that worked (an empty list means none is configured):
```sh
ssh $MAC 'cd ~/flyrely-api && venv/bin/python -c "import alerts, dotenv; dotenv.load_dotenv(\".env.alerts\"); print(alerts.send_alert(\"FlyRely test\", \"spare Mac alert test\"))"'
```

## 6. Offline checks (still no SWIM connection)

```sh
ssh $MAC 'cd ~/flyrely-api && venv/bin/python -m swim.consumer --source tfms --check && venv/bin/python -m swim.consumer --source sfdps --check'
```
Every line must say `OK` (it checks the settings exist, `tcps://`, every Solace import the consumer uses, the trust store
directory, the codec, a writable archive directory). A `FAIL solace-pubsubplus import` on macOS 10.15 means 1.6.0 lacks an API
the consumer needs: stop and report it.

The trust store: on macOS the consumer splits certifi's CA bundle into a private temp directory once per process
(removed at exit), because python.org Python has no system PEM directory. To use another directory set `SCDS_TRUST_STORE_DIR`.

## 7. Install and load the services

Default to a **LaunchDaemon** (starts at boot, needs no login, runs as Sean's user). Preview first, then install.
```sh
ssh $MAC 'cd ~/flyrely-api && venv/bin/python deploy/install_services.py --mode daemon --user $(id -un) --dry-run'
ssh -t $MAC 'cd ~/flyrely-api && sudo venv/bin/python deploy/install_services.py --mode daemon --user $(id -un)'   # [SEAN TTY]
```
This writes `/Library/LaunchDaemons/com.flyrely.swim.{tfms,sfdps,health,retention}.plist` (root-owned, 644) and loads
them: one consumer process per source, a health check every 5 minutes, retention daily at 04:17. Re-running is safe:
unchanged files are left alone and their services are not restarted; `--uninstall` removes them (never touches `data/`).
LaunchAgent alternative (only if the Mac auto-logs-in; no sudo): `--mode agent`.

Behaviour to rely on: `KeepAlive` with `SuccessfulExit=false` restarts a crashed consumer, at most once per 60 s
(`ThrottleInterval`). An authentication failure makes the consumer write a park marker, send one alert, and exit **0**, so
launchd does not respawn it; on every later start it refuses to connect. After fixing the credentials:
```sh
ssh $MAC 'cd ~/flyrely-api && venv/bin/python -m swim.health --unpark tfms'
ssh -t $MAC 'sudo launchctl kickstart -k system/com.flyrely.swim.tfms'     # [SEAN TTY]
```

Verify (give it two minutes):
```sh
ssh $MAC 'cd ~/flyrely-api && venv/bin/python -m swim.health --status && venv/bin/python -m swim.health --dry-run'
ssh $MAC 'cd ~/flyrely-api && ls -la data/swim/tfms/*/ | tail -3; df -h . | tail -1; tail -5 logs/swim-tfms.log'
```
Expect `connected` for both, a few hundred messages per minute received, and `healthy`.

## 8. Reboot test

```sh
ssh -t $MAC 'sudo shutdown -r now'          # [SEAN TTY]
# wait about 3 minutes, no login at the console
until ssh -o ConnectTimeout=5 $MAC true 2>/dev/null; do sleep 15; done
ssh $MAC 'uptime; cd ~/flyrely-api && venv/bin/python -m swim.health --status && venv/bin/python -m swim.health --dry-run'
```
Pass = within about 2 minutes of boot both sources show `connected`, new archive files exist (a new run id per start),
and `health --dry-run` prints `healthy`. Then Sean pulls the power for 10 seconds and the same check is repeated: this
proves "Start up automatically after a power failure". Finally check crash safety:
`ssh $MAC 'pkill -9 -f "swim.consumer --source tfms"'`, wait 90 s, expect a restart with a new archive file while the old file
is still readable (`ssh $MAC 'cd ~/flyrely-api && venv/bin/python -m swim.inspect_tfms --source tfms | head -3'`).

## 9. Day-to-day

- Status: `python -m swim.health --status` (files in `data/swim/status/`). Logs: `~/flyrely-api/logs/` (rotating, 10 MB x 5).
- Retention: `python -m swim.retention --days 14 --dry-run` previews; it deletes only archive files in day folders older than
  14 days and never the newest day of a source.
- Disk: the health check and retention alert below `SWIM_MIN_FREE_GB` (default 10).
- Updating code: repeat the `git archive | tar -x` of step 3 (it does not touch `.env`, `.env.alerts`, `data/` or `venv/`), then
  `sudo launchctl kickstart -k system/com.flyrely.swim.<source>`. After changing `deploy/templates/`, re-run the installer.
- Filter changes: set `SWIM_FILTER_CONFIG` (see `deploy/filter.example.json`), or edit the defaults in `swim/filters.py`.
- LADD list: `deploy/LADD_BLOCKLIST.md`.

## Linux fallback (Oracle Always Free VM, Ubuntu)

Same code, same filters, systemd instead of launchd. Both architectures (x86_64 and the A1 aarch64) have wheels for
`solace-pubsubplus` 1.11.0.
```sh
sudo apt-get update && sudo apt-get install -y python3-venv ca-certificates
mkdir -p ~/flyrely-api    # copy the code as in step 3, then the SCDS_ lines as in step 5
cd ~/flyrely-api && python3 -m venv venv && venv/bin/pip install -r requirements-swim-minimal.txt
venv/bin/python -m swim.consumer --source tfms --check
sudo venv/bin/python deploy/install_services.py --platform linux --mode system --user "$(id -un)" --dry-run
sudo venv/bin/python deploy/install_services.py --platform linux --mode system --user "$(id -un)"
systemctl status flyrely-swim-tfms flyrely-swim-sfdps; systemctl list-timers 'flyrely-swim-*'
```
Trust store: on Linux the consumer uses the system directory `/etc/ssl/certs` automatically (the units also set
`SCDS_TRUST_STORE_DIR=/etc/ssl/certs`); the `ca-certificates` package must be installed. The units use `Restart=on-failure`,
`RestartSec=15`, a start limit of 20 per 15 minutes, and `RestartPreventExitStatus=2 3`: exit 3 (authentication failure) and
2 (configuration error) are never restarted. Unpark with `python -m swim.health --unpark <source>`, then
`sudo systemctl restart flyrely-swim-<source>`. For user-level units without sudo use `--mode user` plus
`sudo loginctl enable-linger $(id -un)`. Reboot test: `sudo reboot`, then the checks of step 8.
