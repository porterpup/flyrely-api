#!/usr/bin/env python3
"""Install the Mac mini LaunchAgents: API, web, three SWIM consumers, retention, health (log only), and the
L1 linkage extractor (com.flyrely.linkage.extract).
No sudo. Idempotent: it writes the plists and (re)loads them. The copilot watcher is NOT installed.

  python deploy/macmini/install_macmini.py [--dry-run] [--uninstall] [--only schedule-refresh]
Redeploy after a code change: see HANDOFF.md ("Hosting: Mac mini").
"""
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LA = Path.home() / "Library" / "LaunchAgents"
LOGS = Path.home() / "Library" / "Logs" / "flyrely"
DOM = f"gui/{os.getuid()}"
SERVICES = {"com.flyrely.api": "run-api.sh", "com.flyrely.web": "run-web.sh"}
LINKAGE = ["com.flyrely.linkage.extract"]
REFRESH = "com.flyrely.schedule.refresh"
SWIM = ["com.flyrely.swim.tfms", "com.flyrely.swim.sfdps", "com.flyrely.swim.stdds",
        "com.flyrely.swim.retention", "com.flyrely.swim.health"]

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key><array><string>/bin/bash</string><string>{script}</string></array>
    <key>WorkingDirectory</key><string>{repo}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>10</integer>
    <key>StandardOutPath</key><string>{logs}/{name}.log</string>
    <key>StandardErrorPath</key><string>{logs}/{name}.log</string>
</dict>
</plist>
"""


def sh(cmd, dry, ignore=False):
    print("  $ " + " ".join(cmd))
    if not dry:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode and not ignore:
            sys.exit(r.stdout + r.stderr)


def install_linkage(py, dry):
    """L1 extractor: RunAtLoad + KeepAlive, logs in ~/Library/Logs/flyrely/linkage-extract*.log."""
    label = LINKAGE[0]
    vals = {"LABEL": label, "PYTHON": str(py), "REPO": str(REPO), "ENV_FILE": str(REPO / ".env"), "LOG_DIR": str(LOGS)}
    text = (REPO / "deploy" / "templates" / "launchd-linkage.plist.tmpl").read_text()
    for k, v in vals.items():
        text = text.replace(f"__{k}__", escape(v))
    p = LA / f"{label}.plist"
    print(("[dry-run] " if dry else "") + "writing", p)
    if not dry:
        LA.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        p.chmod(0o644)
        sh(["launchctl", "bootout", f"{DOM}/{label}"], dry, ignore=True)
        sh(["launchctl", "enable", f"{DOM}/{label}"], dry)
        sh(["launchctl", "bootstrap", DOM, str(p)], dry)


def install_schedule_refresh(dry):
    """Weekly schedule-refresh job (template deploy/templates/launchd-schedule-refresh.plist.tmpl)."""
    sys.path.insert(0, str(REPO / "deploy"))
    from install_services import fill
    py = REPO / "venv" / "bin" / "python"
    text = fill((REPO / "deploy" / "templates" / "launchd-schedule-refresh.plist.tmpl").read_text(), {
        "LABEL": REFRESH, "PYTHON": escape(str(py)), "REPO": escape(str(REPO)),
        "ENV_FILE": escape(str(REPO / ".env")), "LOG_DIR": escape(str(LOGS))})
    p = LA / f"{REFRESH}.plist"
    print(("[dry-run] " if dry else "") + "writing", p)
    if dry:
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    LA.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    p.chmod(0o644)
    sh(["launchctl", "bootout", f"{DOM}/{REFRESH}"], dry, ignore=True)
    sh(["launchctl", "enable", f"{DOM}/{REFRESH}"], dry)
    sh(["launchctl", "bootstrap", DOM, str(p)], dry)


def main():
    dry, un = "--dry-run" in sys.argv, "--uninstall" in sys.argv
    if "--only" in sys.argv and sys.argv[sys.argv.index("--only") + 1:][:1] == ["schedule-refresh"]:
        if un:
            sh(["launchctl", "bootout", f"{DOM}/{REFRESH}"], dry, ignore=True)
            if not dry:
                (LA / f"{REFRESH}.plist").unlink(missing_ok=True)
        else:
            install_schedule_refresh(dry)
        return
    if un:
        for label in list(SERVICES) + SWIM + LINKAGE:
            sh(["launchctl", "bootout", f"{DOM}/{label}"], dry, ignore=True)
            p = LA / f"{label}.plist"
            print("removing", p)
            if not dry and p.exists():
                p.unlink()
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    for label, script in SERVICES.items():
        p = LA / f"{label}.plist"
        text = PLIST.format(label=label, script=escape(str(HERE / script)), repo=escape(str(REPO)),
                            logs=escape(str(LOGS)), name=label.split(".")[-1])
        print(("[dry-run] " if dry else "") + "writing", p)
        if not dry:
            LA.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            p.chmod(0o644)
            sh(["launchctl", "bootout", f"{DOM}/{label}"], dry, ignore=True)
            sh(["launchctl", "enable", f"{DOM}/{label}"], dry)
            sh(["launchctl", "bootstrap", DOM, str(p)], dry)
    py = REPO / "venv" / "bin" / "python"
    install_linkage(py, dry)
    cmd = [str(py), str(REPO / "deploy" / "install_services.py"), "--mode", "agent",
           "--sources", "tfms,sfdps,stdds", "--python", str(py), "--log-dir", str(LOGS),
           # Per-source days live in swim/retention.py DEFAULT_SOURCE_DAYS and beat --days (7 only
           # applies to unlisted sources): SFDPS 35 d (~0.2 GB/day), TFMS 14 d (~0.9 GB/day), STDDS 14 d
           # (~0.3 GB/day after the receive-time filter drops position reports; 4.3 GB/day unfiltered).
           # About 1.4 GB/day total; steady state ~7 + ~13 + ~4 = ~24 GB, ~27 GB peak with headroom.
           # Sean approved a ~30 GB budget on 2026-11-07 (was 20 GB).
           "--retention-days", "7"]
    if dry:
        cmd.append("--dry-run")
    print("  $ " + " ".join(cmd))
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
