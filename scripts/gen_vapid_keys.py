"""Generate a VAPID key pair for web push. Run ONCE at deploy time, on the host that will hold the secret:

    python scripts/gen_vapid_keys.py

Prints the three variables for the host's .env / secret store (never commit them):
    FLYRELY_COPILOT_VAPID_PUBLIC_KEY   (also served to the browser by GET /copilot/config)
    FLYRELY_COPILOT_VAPID_PRIVATE_KEY  (SECRET)
    FLYRELY_COPILOT_VAPID_SUBJECT      (edit: mailto:you@example.com or the app's https URL)

Changing the public key later invalidates every existing browser subscription.
"""
import base64
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def generate() -> tuple[str, str]:
    """Returns (public, private) as unpadded base64url: the public key is the uncompressed P-256 point
    (what `applicationServerKey` expects), the private key is the raw 32-byte scalar (what pywebpush accepts)."""
    key = ec.generate_private_key(ec.SECP256R1())
    public = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    private = key.private_numbers().private_value.to_bytes(32, "big")
    return _b64(public), _b64(private)


def main() -> int:
    public, private = generate()
    print(f"FLYRELY_COPILOT_VAPID_PUBLIC_KEY={public}")
    print(f"FLYRELY_COPILOT_VAPID_PRIVATE_KEY={private}")
    print("FLYRELY_COPILOT_VAPID_SUBJECT=mailto:CHANGE_ME@example.com")
    print("# Keep the private key secret; do not commit it.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
