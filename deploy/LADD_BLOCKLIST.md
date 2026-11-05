# LADD block list (privacy filter)

FAA terms for SWIM data require suppressing flights on the LADD list (Limiting Aircraft Data Displayed;
the related ICAO-address privacy programme is PIA) before any downstream use or display. The consumer
has the hook; **the list itself is not available yet. Obtaining it is Sean's job.**

## How the hook works
- File: `data/ladd_blocklist.txt` (override with `SWIM_BLOCKLIST_FILE`). If the file does not exist,
  nothing is filtered by it (a no-op) and the consumer logs nothing special.
- Format: one identifier per line, or separated by commas/spaces. `#` starts a comment. Case does not
  matter. An identifier can be a callsign (`UAL123`), a registration (`N750EC`) or a 24-bit ICAO hex
  address (`AA1A70`).
- Changes are picked up within about 60 seconds. No restart is needed.
- Where it applies (it runs before every other filter rule, and also with `SWIM_FILTER=off`):
  - **TFMS and SFDPS flight messages are batches.** The list is applied **per flight element**: a listed flight
    is removed from the batch and the other flights in it are kept (or the whole message is dropped if it was the
    only flight). A flight matches on callsign, registration or hex address, whichever the message carries.
  - **Everything else** (STDDS, messages that cannot be parsed, binary bodies, non-flight SFDPS data, or any
    message when the filter is off and it is not a flight message) gets a **best-effort raw-text scan**: if any
    listed identifier appears anywhere in the text as a whole token, the whole message is dropped. This can
    over-drop (a callsign mentioned in an unrelated field) but should not under-drop.
- Dropped messages are acknowledged to the broker but never written.
- This is a best-effort filter, not a guarantee. See the limits below.

## Limits to be aware of (be honest with the FAA about these)
1. **TFMS messages carry only the callsign** (no registration or hex address). A list that contains only
   registrations or hex addresses cannot match TFMS flights. If the FAA list is registration-based, we
   need a registration-to-callsign mapping, or the TFMS-side flights have to be handled downstream
   after linking them to SFDPS (which does carry registration and hex address).
2. **Flow messages (TMI flight lists) are kept whole.** They list many callsigns, so a blocked callsign
   can still appear inside one. Anything that displays or exports flow data must apply the list too.
3. **Nothing is retroactive.** Data archived before the list was loaded is not scrubbed. When the list first
   arrives, run a one-off scrub of the archive (or delete it; retention removes it after 14 days anyway).
4. The list is itself sensitive. It is in `data/`, which is gitignored. Never commit it.

## What Sean needs to do
Ask the FAA SWIM / SCDS support contact (the address on the approval e-mail), in the same thread as the
approval:
> "How do SCDS subscribers obtain the current LADD (and PIA) block list, how often is it updated, and which
> identifiers does it contain (callsign, registration, ICAO 24-bit address)? We archive TFMS and SFDPS raw
> messages and need to suppress listed flights."

Then save the list as `data/ladd_blocklist.txt` on the machine that runs the consumer (the spare Mac:
`~/flyrely-api/data/ladd_blocklist.txt`), and tell Claude the identifier types it contains so the
filter can be checked against a real sample.
