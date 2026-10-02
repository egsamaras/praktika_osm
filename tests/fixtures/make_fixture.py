"""Generate a synthetic bilingual meeting recording with macOS voices (no real PII).

The meeting is fictional: the "Data team weekly" at Acme Bank, with three invented speakers.
Run on macOS: python3 make_fixture.py [out_dir]
Produces synthetic_meeting.wav (16 kHz mono) and synthetic_meeting.truth.json.
"""

import json, pathlib, subprocess, sys

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
VOICES = {"Layla": "Flo (English (UK))", "Omar": "Daniel", "Khalid": "Majed"}
LINES = [
    (
        "Layla",
        "Good morning everyone. This is the data team weekly on the fifteenth of September. We have three items: the meeting notes trial, the privacy notice, and the reporting server upgrade.",
    ),
    (
        "Omar",
        "Thanks Layla. On the meeting notes trial, the prototype now transcribes English and Arabic locally on the laptop. Nothing leaves our own devices. I propose we start with volunteers from the data team.",
    ),
    (
        "Khalid",
        "شكراً عمر. أنا أؤيد الاقتراح، لكن نحتاج مراجعة أمنية قبل البدء. سأكمل المراجعة الأمنية قبل الخامس والعشرين من سبتمبر.",
    ),
    (
        "Layla",
        "Agreed. So the decision is: the trial starts on the fifth of October, limited to the data team, English and Arabic meetings, and the chair asks everyone for permission to record at the start of each meeting.",
    ),
    (
        "Omar",
        "I will draft the privacy notice in both languages by Thursday the eighteenth of September and send it to the privacy team for comments.",
    ),
    (
        "Khalid",
        "One risk: if the transcripts include customer account numbers, we must mask them before the language model sees anything. The notes should be visible only to the meeting owner.",
    ),
    (
        "Layla",
        "Good point, we add that to the checklist. Open question for next time: how long do we keep the recordings once the notes are signed off? My instinct is to delete them after ten days.",
    ),
    (
        "Omar",
        "On the server upgrade, the cost estimate goes to the operations committee on the sixth of October. I will circulate the final numbers on Monday the twenty first.",
    ),
    (
        "Layla",
        "Thank you. To summarise: trial approved from the fifth of October, Omar drafts the notice by the eighteenth, Khalid completes the security review by the twenty fifth, and the retention question comes back next week. Meeting closed.",
    ),
]
files = []
for i, (spk, text) in enumerate(LINES):
    a = out / f"line_{i:02d}.aiff"
    f = out / f"line_{i:02d}.wav"
    subprocess.run(["say", "-v", VOICES[spk], "-r", "170", "-o", str(a), text], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(a), "-ar", "16000", "-ac", "1", str(f)],
        check=True,
    )
    a.unlink()
    files.append(f)
gap = out / "gap.wav"
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-t",
        "0.7",
        str(gap),
    ],
    check=True,
)
concat = out / "concat.txt"
concat.write_text("".join(f"file '{f.name}'\nfile '{gap.name}'\n" for f in files))
wav = out / "synthetic_meeting.wav"
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav),
    ],
    check=True,
)
dur = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
    capture_output=True,
    text=True,
).stdout.strip()
for f in files + [gap, concat]:
    f.unlink()
truth = {
    "meeting": {
        "title": "Data team weekly",
        "date": "2026-09-15",
        "language": "en+ar",
        "duration_s": float(dur),
    },
    "speakers": list(VOICES),
    "script": [{"speaker": s, "text": t} for s, t in LINES],
    "expected": {
        "decisions": [
            "Meeting notes trial starts 2026-10-05, limited to the data team, English and Arabic meetings",
            "The chair asks everyone for permission to record at the start of each meeting",
            "Customer account numbers masked before the language model sees transcripts; notes visible only to the meeting owner",
        ],
        "actions": [
            {
                "owner": "Omar",
                "task": "Draft the privacy notice (EN+AR) and send it to the privacy team for comments",
                "due": "2026-09-18",
            },
            {
                "owner": "Khalid",
                "task": "Complete the security review of the trial",
                "due": "2026-09-25",
            },
            {
                "owner": "Omar",
                "task": "Circulate the final server upgrade numbers",
                "due": "2026-09-21",
            },
        ],
        "open_questions": [
            "How long to keep recordings once the notes are signed off (proposal: delete after 10 days)"
        ],
        "risks": ["Transcripts may contain customer account numbers"],
    },
}
(out / "synthetic_meeting.truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2))
print("duration_s", dur)
