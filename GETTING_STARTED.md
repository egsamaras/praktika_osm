# Getting started with Praktika

This guide is for anyone who has found Praktika on GitHub and wants to try it on a Mac, step by
step, without assuming you are a developer. It takes about 30 to 45 minutes, most of it waiting for
downloads. The [README](README.md) has the same steps in more technical detail.

Praktika is not an app you double-click yet. You run it by typing (or pasting) commands into
**Terminal**, the Mac's command window. Every command below can be pasted as it is. Lines in grey
boxes marked `text` show what you should see, not something to type.

## What you need

* A Mac with Apple silicon (M1, M2, M3, M4 or later). To check: Apple menu → **About This Mac**;
  the "Chip" line should say Apple M-something.
* 24 GB of memory is comfortable; 16 GB works, more slowly (see the README's
  [On a 16 GB Mac](README.md#on-a-16-gb-mac)).
* About 15 GB of free disk space.
* An internet connection for the downloads. Once everything is installed, Praktika does not need
  the internet: your meetings never leave your Mac.

Windows is not supported. A Linux server with an NVIDIA GPU also works, but that is a job for an
IT team: see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Step 1: get the files

On the [GitHub page](https://github.com/egsamaras/praktika_osm), click the green **Code** button,
then **Download ZIP**. Your Mac saves `praktika_osm-main.zip` in Downloads; if it does not unzip
itself, double-click it. You now have a folder called `praktika_osm-main` in Downloads.

(If you already use git, `git clone https://github.com/egsamaras/praktika_osm.git` does the same
and makes updates easier. Then use `cd praktika_osm` in Step 2 instead.)

## Step 2: open Terminal and go into the folder

Open Spotlight (⌘ and the space bar), type **Terminal** and press Return. A window with a text
prompt opens. Paste this and press Return:

```bash
cd ~/Downloads/praktika_osm-main
```

Nothing visible happens; Terminal is now "inside" the Praktika folder. Every later command
assumes you are in this folder. If you close Terminal, open it again and repeat this step.

## Step 3: install the tools (once)

Praktika needs three free tools: **uv** (installs Praktika and its Python libraries), **ffmpeg**
(reads audio files) and **Ollama** (runs the AI model that drafts the minutes). The easiest way to
install them is **Homebrew**, the standard installer for Mac tools.

If you do not have Homebrew, go to [brew.sh](https://brew.sh), copy the one-line command at the
top of the page, paste it into Terminal and follow what it says (it asks for your Mac password;
nothing is shown while you type it). When it finishes, it prints two or three commands under
"Next steps" to add Homebrew to your path: paste those too.

Then install the three tools and start Ollama:

```bash
brew install uv ffmpeg ollama
```

```bash
brew services start ollama
```

## Step 4: download the AI models (once)

The drafting model, Qwen2.5 14B, is about 9 GB:

```bash
ollama pull qwen2.5:14b
```

Then install Praktika itself and download the speech-to-text model, Whisper (about 1.6 GB). The
first command takes a few minutes the first time:

```bash
uv sync --frozen --extra mac
```

```bash
uv run praktika models pull stt_en
```

## Step 5: check your Mac

```bash
uv run praktika doctor
```

Doctor runs 13 checks and prints one line for each, marked OK, WARN or FAIL. On a fresh Mac, a few
WARN lines are normal: `env_file`, `data_dir`, `identity`, `audit_chain` and `endpoint_agent`.
What matters is that no line says **FAIL**. The usual failures:

* `ollama` FAIL: Ollama is not running, or the model is missing. Repeat the two Ollama commands
  above.
* `disk_encryption` FAIL: FileVault is off. Turn it on in System Settings → Privacy & Security →
  FileVault; meeting content should not sit on an unencrypted disk.

## Step 6: your first minutes, from a sample meeting

The folder includes a short, made-up meeting (a bank's working group discussing a new online
account form). Turn it into minutes:

```bash
uv run praktika ingest docs/examples/demo_meeting.vtt --title "My first Praktika meeting"
```

Praktika first asks a few questions, because it never processes a meeting without them. For this
sample, answer as if it were a real meeting you ran:

* "Were all attendees notified?": **y**
* "Did anyone object?": **n**
* "How was notice given?": **chat**
* the question about Teams transcription: **y**
* "Purpose of the recording": a short sentence of at least 10 characters, for example
  `Trying Praktika on the sample meeting`
* then a short list of confirmations that the meeting is in scope (not a board meeting, not about
  HR, no customers present, and so on), each shown in English and Arabic: **y** to each.

Then it drafts. On a laptop this takes a few minutes, during which nothing seems to happen; that is
normal. At the end you see something like:

```text
Draft v1 ready for M-20261002-911a: 3 decisions, 3 actions, 4 flags (0 blocking approval).
```

## Step 7: review and approve in your browser

Start the review page:

```bash
uv run praktika serve
```

It prints a line starting `Review UI on http://127.0.0.1:8793/?t=`. Hold ⌘ and click that link, or
copy all of it into your browser, **including** the part after `?t=`: that is the key to the page.
The page runs on your own Mac; it is not a website on the internet.

What you see:

* **Reviewer flags** at the top: things the checker wants a person to confirm, such as a number or
  a name it could not verify. Click **Mark resolved** once you have checked one.
* **Minutes** on the left: summary, decisions, actions, open questions and risks. Every item shows
  the transcript line it came from (for example `S0011 00:03:17`). For each item, pick a reason
  from its list and press **Accept**, **Save change** (after correcting the text) or **Reject**
  (for example with the reason `not_said`).
* **Transcript** on the right, so you can check each item against what was said.

When you are satisfied, choose a reason at the top (for example `accurate`) and press
**Approve**. Then **Export .docx** gives you the minutes as a Word file (**Export .md** as plain
text). Before approval, export is refused on purpose.

To stop the review page, click in Terminal and press Control and C together.

## Step 8: use it on your own meetings

There are three ways in.

**A Microsoft Teams meeting (the best route).** If the meeting had transcription switched on, open
the meeting's chat in Teams after it ends, go to the **Recap** tab, find the transcript and choose
**Download** (`.vtt` or `.docx`). Whether you can download it depends on your organisation's
Teams settings and on your role in the meeting. Then, in Terminal, in the Praktika folder:

```bash
uv run praktika ingest ~/Downloads/your-meeting.vtt --title "Weekly team meeting"
```

Teams already knows who said what, so the minutes name the right people.

**A recording you already have** (`.m4a`, `.mp3`, `.wav` or `.mp4`):

```bash
uv run praktika ingest ~/Downloads/your-recording.m4a --title "Project meeting"
```

Praktika transcribes it first, which takes longer. A recording does not say who is speaking, so
every line shows the speaker as "unknown" and the minutes cannot tell who said what: add the
owners of actions yourself when you review them.

**A meeting in the room, recorded by your Mac's microphone:**

```bash
uv run praktika start --title "Planning meeting"
```

It shows a short consent script to read out, asks the same questions, then records until you press
Control and C. Every line is labelled `ME`, because one microphone cannot tell voices apart. The first time, macOS asks whether Terminal may use the microphone: allow it. It
hears what your Mac's microphone hears; on a Teams or Zoom call that is your own voice, and the
others only through your speakers. For calls, the Teams transcript route above is better.

After each one, run `uv run praktika serve` again to review, approve and export.

## Good to know

* **It asks for consent every time.** Praktika will not process a meeting unless you confirm that
  people were told and nobody objected. If someone objects, do not use it for that meeting.
* **It starts in pilot mode.** Meetings are treated as Internal, and board, HR, customer, regulator
  and legally privileged meetings are out of scope. That is deliberate for a first release.
* **Where your data is.** Meetings are stored in `~/Library/Application Support/Praktika`, and the
  key that protects personal details in them is in your macOS Keychain, as an item called
  `praktika-vault`. The first time, macOS may ask whether to allow access to it: choose
  **Always Allow**.
* **Deleting old meetings automatically.** Recordings, transcripts and drafts are deleted on a
  schedule once their time is up. To make that happen hourly, run `uv run praktika retention
  install` once and then the `launchctl` command it prints.
* **Removing Praktika.** See [Uninstalling](README.md#uninstalling) in the README.

## When something goes wrong

| What you see | What to do |
|---|---|
| `zsh: command not found: brew` | Homebrew is not installed, or its "Next steps" commands were not run. See Step 3. |
| `zsh: command not found: uv` (or `ollama`, `ffmpeg`) | Run Step 3 again. |
| `cd: no such file or directory` | The folder has a different name or place. In Finder, drag the Praktika folder onto the Terminal window after typing `cd ` (with a space), then press Return. |
| The draft seems stuck | It is probably still drafting; a long meeting can take 10 minutes or more on a laptop. If nothing changes for much longer, check that Ollama is running (Step 5). |
| The review page says the link is invalid or refuses access | Open the link exactly as printed, including everything after `?t=`. |
| `error: ... refused` when you ingest | One of your answers stopped the meeting, for example an objection or an out-of-scope meeting. That is the consent gate working. |
| Export is refused | The minutes are not approved yet. Approve them on the review page first. |

## Getting help

Open an issue on [GitHub](https://github.com/egsamaras/praktika_osm/issues). Please **never**
include real meeting content, recordings, names or other personal details: describe the problem
with the sample meeting instead.
