#!/usr/bin/env python3
"""Install the Mac mini LaunchAgents: API, web, three SWIM consumers, retention, health (log only).
No sudo. Idempotent: it writes the plists and (re)loads them. The copilot watcher is NOT installed.

  python deploy/macmini/install_macmini.py [--dry-run] [--uninstall]
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


def main():
    dry, un = "--dry-run" in sys.argv, "--uninstall" in sys.argv
    if un:
        for label in list(SERVICES) + SWIM:
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
    cmd = [str(py), str(REPO / "deploy" / "install_services.py"), "--mode", "agent",
           "--sources", "tfms,sfdps,stdds", "--python", str(py), "--log-dir", str(LOGS),
           # 7 days of TFMS+SFDPS (~1.1 GB/day) plus STDDS 1 day (~4.3 GB/day, set in swim/retention.py)
           # keeps the archive near 16 GB, under the 20 GB budget
           "--retention-days", "7"]
    if dry:
        cmd.append("--dry-run")
    print("  $ " + " ".join(cmd))
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
