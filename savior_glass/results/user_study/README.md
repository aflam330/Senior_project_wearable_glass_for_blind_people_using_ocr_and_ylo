# Voice user study: responses without names

`voice_responses_anonymized.csv` is `study_data/voice/responses.csv` with the `name` column removed: 103 rows,
six yes / no questions (easy_to_use, speech_clear, helped_currency, helped_reading, trust, would_use_daily).

`source` says where a row comes from:
- `app` (38 rows, P001-P040, 18 November 2026): recorded by the glass app on the Raspberry Pi 5.
- `imported` (65 rows, P041-P105, 6-9 October 2026): collected on the laptop and added from a text file. There are no recordings or run logs for these rows.

The names, voice recordings and camera photos stay in `study_data/` on the device and are not published.
Regenerate this file after new sessions: drop the `name` column from `study_data/voice/responses.csv`.
