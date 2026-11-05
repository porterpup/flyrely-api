#!/usr/bin/env python3
"""Install (or preview) the FlyRely SWIM services: one consumer per source, a health check and a
retention job. Standard library only; works on the python.org Python 3.9+ and on Linux.

  macOS  --mode agent   LaunchAgent in ~/Library/LaunchAgents (runs while that user is logged in; needs
                        auto-login on the spare Mac)
         --mode daemon  LaunchDaemon in /Library/LaunchDaemons with UserName=<user> (starts at boot,
                        no login needed; run with sudo)
  Linux  --mode system  /etc/systemd/system units with User=<user> (run with sudo)
         --mode user    ~/.config/systemd/user units (needs `loginctl enable-linger <user>`)

Idempotent: files identical to what is already installed are left alone and their service is not
restarted; changed files are rewritten and reloaded. --dry-run prints every file and command and
changes nothing. --uninstall stops and removes the services (never touches data/).

Examples
  python deploy/install_services.py --dry-run
  python deploy/install_services.py --mode daemon --user seans --sources tfms,sfdps     (via sudo)
"""
import argparse
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMPLATES = HERE / "templates"
LABEL_PREFIX = "com.flyrely.swim"


def fill(text: str, values: dict) -> str:
    for k, v in values.items():
        text = text.replace(f"__{k}__", v)
    left = re.findall(r"__[A-Z_]+__", text)
    if left:
        raise ValueError(f"unfilled placeholders: {sorted(set(left))}")
    return text


def detect_platform() -> str:
    return "macos" if platform.system() == "Darwin" else "linux"


def build(opts) -> list:
    """Returns [(kind, name, path, content)] for every file to install."""
    plat, mode = opts.platform, opts.mode
    base = {
        "PYTHON": str(opts.python), "REPO": str(opts.repo), "ENV_FILE": str(opts.env_file),
        "LOG_DIR": str(opts.log_dir), "SOURCES": ",".join(opts.sources),
        "RETENTION_DAYS": str(opts.retention_days), "MIN_FREE_GB": f"{opts.min_free_gb:g}",
    }
    files = []
    if plat == "macos":
        if mode == "agent":
            d = Path(opts.out_dir) if opts.out_dir else Path.home() / "Library" / "LaunchAgents"
            daemon_keys = ""
        else:
            d = Path(opts.out_dir) if opts.out_dir else Path("/Library/LaunchDaemons")
            daemon_keys = (f"    <key>UserName</key>\n    <string>{escape(opts.user)}</string>\n")
        esc = {k: escape(v) for k, v in base.items()}

        def plist(tmpl, label, **extra):
            vals = {**esc, "LABEL": label, "DAEMON_KEYS": daemon_keys, **{k: escape(v) for k, v in extra.items()}}
            return fill((TEMPLATES / tmpl).read_text(), vals)
        for s in opts.sources:
            label = f"{LABEL_PREFIX}.{s}"
            files.append(("service", label, d / f"{label}.plist", plist("launchd-consumer.plist.tmpl", label, SOURCE=s)))
        for kind, tmpl in (("health", "launchd-health.plist.tmpl"), ("retention", "launchd-retention.plist.tmpl")):
            label = f"{LABEL_PREFIX}.{kind}"
            files.append(("job", label, d / f"{label}.plist", plist(tmpl, label)))
    else:
        sd = lambda v: v.replace("%", "%%")  # systemd specifier escape; quotes are in the templates
        base = {k: sd(v) for k, v in base.items()}
        if mode == "system":
            d = Path(opts.out_dir) if opts.out_dir else Path("/etc/systemd/system")
            user_line, wanted = f"User={opts.user}\n", "multi-user.target"
        else:
            d = Path(opts.out_dir) if opts.out_dir else Path.home() / ".config" / "systemd" / "user"
            user_line, wanted = "", "default.target"
        vals = {**base, "USER_LINE": user_line, "WANTED_BY": wanted}
        for s in opts.sources:
            name = f"flyrely-swim-{s}.service"
            files.append(("service", name, d / name,
                          fill((TEMPLATES / "systemd-consumer.service.tmpl").read_text(), {**vals, "SOURCE": s})))
        for kind in ("health", "retention"):
            for ext in ("service", "timer"):
                name = f"flyrely-swim-{kind}.{ext}"
                files.append(("timer" if ext == "timer" else "job-service", name, d / name,
                              fill((TEMPLATES / f"systemd-{kind}.{ext}.tmpl").read_text(), vals)))
    return files


def run(cmd, dry, check=False):
    print("  $ " + " ".join(cmd))
    if dry:
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode and check:
        print(r.stdout + r.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.returncode


def launchd_domain(opts) -> str:
    return "system" if opts.mode == "daemon" else f"gui/{os.getuid()}"


def wait_unloaded(dom, label, dry, tries=30):
    """bootout can return before the old process has exited; starting the new one too early fails or
    leaves two consumers on one queue. Poll until launchd no longer knows the job."""
    if dry:
        print(f"  (would wait up to {tries}s until {dom}/{label} is unloaded)")
        return
    for _ in range(tries):
        if subprocess.run(["launchctl", "print", f"{dom}/{label}"], capture_output=True).returncode != 0:
            return
        time.sleep(1)
    raise SystemExit(f"{label} is still loaded {tries}s after bootout; not starting a second copy")


def activate(opts, files, changed, existing_changed, dry):
    if opts.no_load:
        print("--no-load: not loading anything")
        return
    if opts.platform == "macos":
        dom = launchd_domain(opts)
        for kind, label, path, _ in files:
            if dry:
                if label in changed:
                    print(f"  {label}: would be (re)loaded")
                else:
                    print(f"  {label}: unchanged; at run time it is loaded only if launchctl does not already have it")
                run(["launchctl", "enable", f"{dom}/{label}"], dry)
                run(["launchctl", "bootstrap", dom, str(path)], dry)
                continue
            loaded = run(["launchctl", "print", f"{dom}/{label}"], dry) == 0
            if loaded and label not in changed:
                print(f"  {label}: already loaded and unchanged")
                continue
            if loaded:
                run(["launchctl", "bootout", f"{dom}/{label}"], dry)
                wait_unloaded(dom, label, dry)
            run(["launchctl", "enable", f"{dom}/{label}"], dry)  # enable first, so bootstrap cannot be refused
            run(["launchctl", "bootstrap", dom, str(path)], dry, check=True)
    else:
        sc = ["systemctl"] + (["--user"] if opts.mode == "user" else [])
        run(sc + ["daemon-reload"], dry, check=True)
        for kind, name, _, _ in files:
            if kind in ("service", "timer"):
                run(sc + ["enable", "--now", name], dry, check=True)
                if name in existing_changed and kind == "service":
                    run(sc + ["restart", name], dry, check=True)


def install(opts) -> int:
    files = build(opts)
    dry = opts.dry_run
    changed, existing_changed = set(), set()
    for kind, name, path, content in files:
        try:
            same = path.exists() and path.read_text() == content
        except OSError:
            same = False
        state = "unchanged" if same else ("new" if not path.exists() else "changed")
        print(f"{'[dry-run] ' if dry else ''}{state:9s} {path}")
        if dry and opts.show:
            print("-" * 60 + "\n" + content + "-" * 60)
        if not same:
            changed.add(name)
            if state == "changed":
                existing_changed.add(name)
            if not dry:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                if opts.platform == "macos":
                    path.chmod(0o644)  # launchd rejects group/world-writable plists
    if not dry:
        Path(opts.log_dir).mkdir(parents=True, exist_ok=True)
        if opts.mode in ("daemon", "system") and os.geteuid() == 0:
            import shutil
            shutil.chown(str(opts.log_dir), user=opts.user)  # the service runs as --user and writes logs here
    activate(opts, files, changed, existing_changed, dry)
    print("done" if not dry else "dry run complete: nothing was changed")
    return 0


def uninstall(opts) -> int:
    files = build(opts)
    dry = opts.dry_run
    for kind, name, path, _ in files:
        if opts.platform == "macos":
            run(["launchctl", "bootout", f"{launchd_domain(opts)}/{name}"], dry)
        elif kind in ("service", "timer"):
            run(["systemctl"] + (["--user"] if opts.mode == "user" else []) + ["disable", "--now", name], dry)
        print(f"{'would remove' if dry else 'removing'} {path}")
        if not dry and path.exists():
            path.unlink()
    if opts.platform == "linux":
        run(["systemctl"] + (["--user"] if opts.mode == "user" else []) + ["daemon-reload"], dry)
    return 0


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", choices=("macos", "linux"), default=detect_platform())
    ap.add_argument("--mode", choices=("agent", "daemon", "system", "user"), default=None)
    ap.add_argument("--sources", default="tfms,sfdps", help="comma separated; one process per source")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--python", type=Path, default=Path(os.path.abspath(sys.executable)),
                    help="interpreter that has the requirements-swim.txt packages (default: this one)")
    ap.add_argument("--env-file", type=Path, default=None, help="default <repo>/.env")
    ap.add_argument("--log-dir", type=Path, default=None, help="default <repo>/logs")
    ap.add_argument("--user", default=os.environ.get("SUDO_USER") or os.environ.get("USER", ""))
    ap.add_argument("--retention-days", type=int, default=14)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, default=None, help="write the service files here instead (tests)")
    ap.add_argument("--no-load", action="store_true", help="write files but do not load/enable them")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen; change nothing")
    ap.add_argument("--show", action="store_true", help="with --dry-run, also print the rendered files")
    ap.add_argument("--uninstall", action="store_true")
    o = ap.parse_args(argv)
    o.mode = o.mode or ("agent" if o.platform == "macos" else "system")
    if (o.platform == "macos") != (o.mode in ("agent", "daemon")):
        ap.error(f"--mode {o.mode} does not fit --platform {o.platform}")
    o.sources = [s.strip() for s in o.sources.split(",") if s.strip()]
    bad = [s for s in o.sources if s not in ("tfms", "sfdps", "stdds")]
    if bad or not o.sources:
        ap.error("--sources must be a comma separated subset of tfms,sfdps,stdds")
    o.repo = o.repo.resolve()
    o.env_file = o.env_file or o.repo / ".env"
    o.log_dir = o.log_dir or o.repo / "logs"
    for v in (o.python, o.repo, o.env_file, o.log_dir):
        if any(ch in str(v) for ch in '"\n'):
            ap.error("paths must not contain double quotes or newlines")
    if o.retention_days < 1:
        ap.error("--retention-days must be at least 1")
    if (o.mode in ("daemon", "system")) and not o.user:
        ap.error("--user is required for daemon/system mode")
    return o


def main(argv=None) -> int:
    o = parse(argv)
    return uninstall(o) if o.uninstall else install(o)


if __name__ == "__main__":
    sys.exit(main())
