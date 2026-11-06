# Airline contacts: pre-launch hand check

Launch checklist item. Each carrier below is shown in the app but has status `official_source_unconfirmed` in `airlines.json`. Call the number or open the URL, then set `verification` to `verified` and `verified_on` to the date.

| Carrier | Phone shown | Manage URL shown | Check by hand | Done |
|---|---|---|---|---|
| AA American | 800-433-7300 | https://www.aa.com/reservation/view/find-your-reservation | Call: reaches AA reservations. Open URL: find-trip form loads. (aa.com blocks bots, 403.) | [ ] |
| UA United | 800-864-8331 (1-800-UNITED-1) | https://www.united.com/en/us/manageres/mytrips | Call: reaches reservations/customer care. Open URL: find-trip form loads. Confirm text line UNITED to 32050. | [ ] |
| B6 JetBlue | 800-538-2583 (1-800-JETBLUE) | https://www.jetblue.com/manage-trips (already read directly) | Call: reaches customer support, not groups desk. | [ ] |

Already verified by direct read (no action): DL, WN, AS, F9, HA, G4, SY. NK (Spirit) shows no phone; re-check `status_url` is still current.
Sun Country is being acquired by Allegiant; re-check G4/SY before launch.
