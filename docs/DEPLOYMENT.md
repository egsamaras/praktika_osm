# Deploying on a Linux GPU server

This runbook installs Praktika on one Linux server with an NVIDIA GPU: Praktika in a Python virtual
environment, English speech-to-text from Whisper large-v3-turbo served by vLLM in a container,
drafting by `qwen2.5:14b` in Ollama, and the review page on loopback. Nothing in it needs the
server to reach the internet: everything can be prepared on a connected build machine and carried
across. It has been written for Ubuntu 24.04; the two hardware examples used for sizing are an
NVIDIA DGX Spark (GB10, aarch64, 128 GB of memory shared between CPU and GPU) and a server with
one NVIDIA L40S (x86_64, 48 GB).

Read [LIMITATIONS.md](LIMITATIONS.md) first. This runbook gives you a single-host installation for
a small group of users who reach the review page over SSH; it is not a highly available service.

Conventions: lines marked **[root]** run in a root shell. Lines that start with `sudo -u praktika`
run as the service account. `<...>` marks a value you supply.

## 1. What you end up with

| Item | Value |
|---|---|
| Service account | `praktika` (no login shell); operators are members of its group |
| Checkout and virtual environment | `/opt/praktika`, `/opt/praktika/.venv` |
| Data directory | `/var/lib/praktika` (mode 0700): database, retained audio, exports, audit log |
| Environment file | `/etc/praktika/praktika.env` (mode 0640); secrets never go in it |
| Model weights and register | `/srv/praktika-models` |
| Speech server | vLLM container on `127.0.0.1:8801`, served model name `whisper-large-v3` |
| Drafting model | Ollama on `127.0.0.1:11434`, model `qwen2.5:14b` |
| Review page | `praktika serve` on `127.0.0.1:8793`, reached through an SSH tunnel |
| Retention | systemd timer `praktika-retention.timer`, hourly |

## 2. Host requirements

| Item | Requirement |
|---|---|
| Operating system | Linux on x86_64 or aarch64 with glibc 2.28 or later. The commands assume Ubuntu 24.04 |
| Python | 3.12 (`python3.12` and `python3.12-venv`); the project requires `>=3.12,<3.13` |
| GPU stack | An NVIDIA driver supported by the vLLM image you choose, Docker, and the NVIDIA Container Toolkit |
| Other packages | `git`, `ffmpeg` (needed for audio ingest only) |
| Host memory | 16 GiB is the floor `doctor` checks; with a discrete GPU, 64 GB or more |
| Disk | Allow about 200 GB across Docker's data root (the vLLM image is tens of GB), `/opt`, `/srv`, Ollama's store and the data directory |
| Data directory | On a local filesystem (the audit log's lock is an `flock`, which does not work over NFS) and on an encrypted volume |
| Network | No internet route needed at run time. Block outbound traffic for the host, the Ollama service and the containers: Praktika's own allow-list governs only Praktika's process |
| Clock | Synchronised (NTP). Audit ordering, retention timers and OIDC expiry checks depend on it |

### GPU memory

vLLM pre-allocates a fraction of the GPU's **total** memory, whatever the model needs. Never leave
`--gpu-memory-utilization` at its default (about 0.9): it would leave nothing for the drafting
model. Ollama allocates the model's weights plus a key-value cache for the context Praktika asks
for, once per parallel slot. For `qwen2.5:14b` (48 layers, 8 key-value heads, head dimension 128)
the cache is 2 × 48 × 8 × 128 × 2 bytes = 196,608 bytes per token, exactly 6 GiB at 32,768 tokens.

All figures in the table are in GiB (1 GiB = 1,073,741,824 bytes; 9 GB is about 8.4 GiB).

| | DGX Spark (128 GB of unified memory) | One L40S (48 GB; about 45 GiB usable with ECC on) |
|---|---|---|
| vLLM `--gpu-memory-utilization` | `0.12`: about 15 reserved | `0.20`: about 9 reserved |
| Whisper large-v3-turbo weights | about 1.5, inside the reservation | about 1.5, inside the reservation |
| `qwen2.5:14b` (Q4_K_M) weights | about 8.4 | about 8.4 |
| Its key-value cache at 32,768 tokens | 6 | 6 |
| Ollama in all, while loaded (weights, cache and working buffers) | about 15 to 17 | about 15 to 17 |
| **Total in use** (vLLM reservation plus Ollama) | **about 30 to 32** | **about 24 to 26** |

On unified memory (the DGX Spark) the GPU's memory is the system's memory: a fraction that is too
large starves the operating system, and a competing workload can get a container killed. On a
discrete GPU, start the speech server before the drafting model is loaded (section 7).

These are planning figures, not measurements of this release on that hardware.

## 3. Service account and directories

```bash
# [root]
id -u praktika >/dev/null 2>&1 || useradd --system --home-dir /var/lib/praktika --shell /usr/sbin/nologin praktika
usermod -aG praktika <operator>                       # once per operator
install -d -m 0700 -o praktika -g praktika /var/lib/praktika
install -d -m 0750 -o praktika -g praktika /etc/praktika
install -m 0640 -o praktika -g praktika /dev/null /etc/praktika/praktika.env
install -d -m 0750 -o praktika -g praktika /opt/praktika
install -d -m 0755 -o praktika -g praktika /srv/praktika-models
apt-get install -y git python3.12 python3.12-venv ffmpeg
```

Every Praktika command then runs as the service account:

```bash
sudo -u praktika -E /opt/praktika/.venv/bin/praktika <command> ...
```

Every command that opens the meeting store (`ingest`, `start`, `transcribe`, `generate`, `abort`,
`approve`, `export`, `serve`, `search`, `actions`, `hold`, `dsar`, `retention run`,
`audit verify`) forces the data directory to mode 0700 and stops if another account owns it, so
running as the service account is not optional. `-E` carries `PRAKTIKA_ENV_FILE` and
`PRAKTIKA_VAULT_KEY` across `sudo`; the sudoers rule that lets operators run commands as
`praktika` must carry the `SETENV` tag, or every `-E` command is refused. `-E` also keeps the
operator's own `HOME`, which is why `PRAKTIKA_ENV_FILE` must always be exported (section 5):
without it Praktika looks for its default env file under the operator's home directory and stops
with an error telling you to export the variable.

The audit log's `actor` field names the account the operator logged in with (the kernel login
uid, which `sudo -u` keeps), not `praktika`. Environment variables such as `SUDO_USER` are never
trusted. Note that anyone who can log in to the host and act as `praktika` can read every meeting.

## 4. Install Praktika

Praktika is installed **editable from a checkout that stays in place**: the prompts (`prompts/`),
the glossary (`glossary.yaml`) and the export templates (`templates/render/`) are read from the
checkout, not from the installed package. It is not published on PyPI, and a wheel built from the
repository does not run on its own. Install no optional extras on the server. In particular, do not
install the `cuda` extra: on aarch64 it installs and then runs speech on the CPU only, about thirty
times too slow, without an error, and on x86_64 it would load a second speech model outside the
memory budget above.

### Option A: with uv and a package index (or an internal mirror)

```bash
# [root] uv for every account, from its release archive verified against its checksum
#        (or your package mirror); see https://docs.astral.sh/uv/
sudo -u praktika git clone https://github.com/egsamaras/praktika_osm.git /opt/praktika
sudo -u praktika git -C /opt/praktika checkout --quiet <release-tag>
cd /opt/praktika && sudo -u praktika env UV_PYTHON=/usr/bin/python3.12 UV_PYTHON_DOWNLOADS=never \
    uv sync --frozen --no-cache
sudo -u praktika /opt/praktika/.venv/bin/praktika --help
sudo -u praktika chmod -R go-w /opt/praktika    # only praktika may change the installed code
```

`uv sync --frozen` installs exactly the versions in `uv.lock` into `/opt/praktika/.venv`, with
Praktika itself in editable mode. `--no-cache` keeps uv's cache out of the service account's home,
which is the data directory. On x86_64 the virtual environment takes about 6.6 GB, because PyPI's
`torch` wheel for that architecture brings NVIDIA's CUDA libraries; on aarch64 it is much smaller.

### Option B: offline, from a wheelhouse

On a connected build machine with the **same CPU architecture** and Python 3.12:

```bash
git clone https://github.com/egsamaras/praktika_osm.git && cd praktika_osm
git checkout <release-tag>
uv export --frozen --no-emit-project --format requirements-txt -o requirements.txt
python3.12 -m pip download --only-binary=:all: --require-hashes -r requirements.txt -d wheelhouse
python3.12 -m pip download --only-binary=:all: -d wheelhouse "setuptools>=77"
cp requirements.txt wheelhouse/
git bundle create praktika.bundle --all
sha256sum praktika.bundle wheelhouse/* > SHA256SUMS
```

`requirements.txt` pins every dependency with its SHA-256. On x86_64 the wheelhouse is about
4 GB, because PyPI's `torch` brings its CUDA libraries; Praktika uses `torch` only for voice
activity detection, on the CPU. Carry `praktika.bundle`, `wheelhouse/` and `SHA256SUMS` to a
directory on the server that the `praktika` account can read, here `/srv/praktika-transfer`, then:

```bash
cd /srv/praktika-transfer && sha256sum -c SHA256SUMS          # every line must say OK
sudo -u praktika git clone /srv/praktika-transfer/praktika.bundle /opt/praktika
sudo -u praktika git -C /opt/praktika checkout --quiet <release-tag>
sudo -u praktika python3.12 -m venv /opt/praktika/.venv
sudo -u praktika /opt/praktika/.venv/bin/pip install --no-index \
    --find-links /srv/praktika-transfer/wheelhouse --require-hashes \
    -r /srv/praktika-transfer/wheelhouse/requirements.txt
sudo -u praktika /opt/praktika/.venv/bin/pip install --no-index \
    --find-links /srv/praktika-transfer/wheelhouse "setuptools>=77"
sudo -u praktika /opt/praktika/.venv/bin/pip install --no-index --no-build-isolation --no-deps \
    -e /opt/praktika
sudo -u praktika /opt/praktika/.venv/bin/praktika --help
sudo -u praktika chmod -R go-w /opt/praktika
```

`sudo` gives the service account a umask that leaves new files group-writable, and operators are
in its group; the final `chmod` stops an operator from changing the installed code without an
audit record.

## 5. Vault key and environment file

**The vault key is mandatory on Linux.** Identifiers are tokenised before any model sees the text,
and the token vault is encrypted with this Fernet key. There is no key store to fall back on, so
with the variable unset every ingest fails at the redaction step. Generate it once:

```bash
/opt/praktika/.venv/bin/python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Keep it in a secrets manager or password manager with a named owner and a second holder, and
export it into the environment of every shell or service that ingests or exports with
`--detokenise`. **Never put it in the env file.** Losing it makes de-tokenisation of earlier
meetings impossible; leaking it re-identifies every token in every transcript still on the host.
There is no rotation or re-keying command.

```bash
export PRAKTIKA_VAULT_KEY='<the key>'
```

Write `/etc/praktika/praktika.env` (as `praktika`, or as root keeping owner and mode):

```ini
PRAKTIKA_MODE=local
PRAKTIKA_PILOT=true
PRAKTIKA_DATA_DIR=/var/lib/praktika
PRAKTIKA_MODELS_DIR=/srv/praktika-models
PRAKTIKA_ALLOWED_HOSTS=["localhost","127.0.0.1"]
PRAKTIKA_LLM_PROVIDER=ollama
PRAKTIKA_LLM_BASE_URL=http://127.0.0.1:11434
PRAKTIKA_LLM_MODEL=qwen2.5:14b
PRAKTIKA_LLM_FALLBACK_MODEL=qwen2.5:14b
PRAKTIKA_LLM_NUM_CTX=32768
PRAKTIKA_STT_EN=http
PRAKTIKA_STT_AR=none
PRAKTIKA_STT_HTTP_URL=http://127.0.0.1:8801
PRAKTIKA_DIARIZE=false
PRAKTIKA_DIARIZE_BACKEND=none
PRAKTIKA_REVIEW_HOST=127.0.0.1
PRAKTIKA_REVIEW_PORT=8793
PRAKTIKA_AUDIT_SINK=jsonl
PRAKTIKA_IDENTITY_PROVIDER=session
```

Then, in every shell that runs Praktika (and in any unit file):

```bash
export PRAKTIKA_ENV_FILE=/etc/praktika/praktika.env
```

Why these values:

* `PRAKTIKA_STT_EN=http` sends every speech chunk to the vLLM server. The default (`mlx_whisper`)
  is the in-process Apple-silicon backend.
* `PRAKTIKA_STT_AR=none` keeps the Arabic path off: every chunk is decoded as English and
  `--lang ar-mixed` is refused before the consent script is shown.
* `PRAKTIKA_LLM_FALLBACK_MODEL` is set to the main model on purpose. The default, `llama3.1:8b`, is
  used above 60,000 transcript tokens, is not checked by `doctor`, and is probably not installed.
* `PRAKTIKA_LLM_NUM_CTX=32768` is sent on every Ollama call; Ollama's own default of 2048 would
  silently truncate the transcript.
* `PRAKTIKA_ALLOWED_HOSTS` is the egress allow-list: every configured URL must match it or
  Praktika will not start.

Praktika reads exactly one env file: the one `PRAKTIKA_ENV_FILE` names, or else `.env` in the
account's default data directory (`~/.local/share/praktika/.env`), which `PRAKTIKA_DATA_DIR` does
not move. A `.env` in the working directory is never read. If `PRAKTIKA_ENV_FILE` names a file that
is missing or unreadable, every command refuses to start. An unknown key in the env file is a
start-up error, and a key with an empty value must be commented out; a misspelt variable exported in
the process environment, by contrast, is silently ignored, so check `praktika config show`.
`.env.example` in the repository lists every key with its allowed values and default.

## 6. Speech model weights

On a connected machine with the `huggingface_hub` package installed (it provides the `hf`
command), pin the download to a revision and record it:

```bash
REV=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('openai/whisper-large-v3-turbo').sha)")
hf download openai/whisper-large-v3-turbo --revision "$REV" --local-dir ./whisper-large-v3-turbo
echo "$REV" > ./whisper-large-v3-turbo/REVISION
```

Use the official `openai/whisper-large-v3-turbo` repository here. `praktika models pull stt_en`
mirrors an MLX conversion for Apple silicon, which vLLM cannot serve.

Carry the directory across, then register it so that every draft's provenance records the
revision and the per-file hashes:

```bash
test -e /srv/praktika-models/stt_en || \
    sudo -u praktika cp -R /srv/praktika-transfer/whisper-large-v3-turbo /srv/praktika-models/stt_en
sudo -u praktika chmod -R u=rwX,go=rX /srv/praktika-models/stt_en
sudo -u praktika -E /opt/praktika/.venv/bin/praktika models register stt_en \
    /srv/praktika-models/stt_en --repo openai/whisper-large-v3-turbo \
    --revision "$(cat /srv/praktika-models/stt_en/REVISION)" --licence MIT
sudo -u praktika -E /opt/praktika/.venv/bin/praktika models verify    # exit 1 on any difference
sudo -u praktika chmod -R go-w /srv/praktika-models
```

The copy line is guarded because a second `cp -R` into an existing directory nests a copy inside
it, and registering that records twice the files without an error; check the file count that
`models register` prints against the directory. `--revision` defaults to the word `local`, which
proves nothing, so always pass the recorded revision. The weights must stay readable by every
account, because the speech container reads them under its own user id; they are public weights.

With the speech backend on HTTP, nothing loads these files in Praktika's own process, so
`praktika models verify` is the check that they still match; and the register describes the
directory you registered, not what the server actually loaded.

## 7. Speech server (vLLM)

Bring the vLLM image through your container registry and **run it by digest, never by tag**.
Record the digest and compare it with the publisher's value on a second machine.

```bash
docker run -d --name praktika-stt-en --gpus all --restart unless-stopped \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -p 127.0.0.1:8801:8000 \
  -v /srv/praktika-models/stt_en:/models/stt_en:ro \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  <vllm-image>@<digest> \
  vllm serve /models/stt_en \
    --served-model-name whisper-large-v3 \
    --gpu-memory-utilization <0.12 on a DGX Spark, 0.20 on an L40S>
```

* **`--served-model-name` must be exactly `whisper-large-v3`.** Praktika sends that name, so any
  other name makes every chunk fail with `STT server returned 404`. For the same reason,
  `PRAKTIKA_STT_HTTP_URL` carries no path: the client appends `/v1/audio/transcriptions` itself.
* `VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1` stop vLLM trying to send anonymous usage
  statistics, which it does by default. `HF_HUB_OFFLINE=1` stops it looking for the model online.
* `-p 127.0.0.1:8801:8000` binds loopback only. The server accepts any caller without
  authentication, and Praktika's speech client has no credential field.
* If `--read-only` or `--cap-drop ALL` stops the image from starting, record what you had to relax
  and why; do not drop the hardening silently.
* `--restart unless-stopped` brings the server back after a reboot. Decide whether that is what you
  want: a restarted server resumes processing without anyone checking.
* On a server with more than one GPU, replace `--gpus all` with `--gpus "device=<gpu-uuid>"` and
  give Ollama the same card (section 8). Use UUIDs: CUDA inside Ollama may number cards differently
  from Docker and `nvidia-smi`.
* On a discrete GPU, start this server before the drafting model is loaded, and wait until
  `curl -s http://127.0.0.1:8801/v1/models` lists `whisper-large-v3`. vLLM measures free memory
  while it starts and refuses to start, or miscounts, if another process takes or frees memory
  meanwhile.

**Check that word timestamps come back.** Praktika asks for them (`response_format=verbose_json`,
`timestamp_granularities[]=word`); whether the server returns them depends on the vLLM release and
its options. Without them, Praktika silently falls back to chunk-level timing and every citation
becomes up to 28 seconds wide. Probe with speech, not a tone (for example the synthetic recording
`tests/fixtures/synthetic_meeting.wav`):

```bash
ffmpeg -i <recording-with-speech> -t 20 -ac 1 -ar 16000 /tmp/probe.wav
curl -s -F model=whisper-large-v3 -F response_format=verbose_json \
     -F 'timestamp_granularities[]=word' -F file=@/tmp/probe.wav \
     http://127.0.0.1:8801/v1/audio/transcriptions | head -c 600
```

The response must contain a `words` array. If it does not, check `vllm serve --help` in your image
for a word-timestamp option and record what you find.

## 8. Drafting model (Ollama)

Install Ollama for your architecture from its release archive verified against the published
checksum, or from your package mirror; do not pipe an installer script from the internet into a
shell. Unpack the whole archive, `lib/ollama` included: it holds the GPU runners, and without it
Ollama runs on the CPU and says nothing.

```bash
# [root]
useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
tar -C /usr -xf <ollama-linux-archive>
cat > /etc/systemd/system/ollama.service <<'UNIT'
[Unit]
Description=Ollama
After=network-online.target
[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/praktika.conf <<'DROPIN'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_NUM_PARALLEL=1"
DROPIN
# on a multi-GPU server, add: Environment="CUDA_VISIBLE_DEVICES=<gpu-uuid>"
systemctl daemon-reload && systemctl enable --now ollama
systemctl show ollama -p Environment     # must show the three variables above
```

The drop-in must start with its `[Service]` line; without it systemd ignores every line and says so
only in the journal, which is why the `systemctl show` check matters. If your package installs its
own account and unit, keep them and add only the drop-in.

Get the model onto the server. With network access, `ollama pull qwen2.5:14b`. Without it, pull it
on a connected machine and copy that store's `manifests/` and `blobs/` trees, keeping their folder
structure, into the service's store, which is `/usr/share/ollama/.ollama/models` unless
`systemctl show ollama -p Environment` shows `OLLAMA_MODELS`:

```bash
# [root]
cp -R <carried-store>/manifests <carried-store>/blobs /usr/share/ollama/.ollama/models/
chown -R ollama:ollama /usr/share/ollama/.ollama/models
systemctl restart ollama
ollama list                                                   # expect qwen2.5:14b
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen2.5:14b","keep_alive":"30m","options":{"num_ctx":32768}}'
ollama ps                                                     # expect PROCESSOR 100% GPU
```

Any CPU share in `ollama ps` means the model did not fit in the memory that was free when it
loaded. Notes:

* `OLLAMA_NUM_PARALLEL=1` keeps one context. Praktika makes one model call at a time; each extra
  slot would reserve another 6 GiB of cache for nothing.
* Praktika sends `num_ctx` and `keep_alive` (10 minutes) on every call, so the model stays loaded
  between one meeting's calls and leaves memory 10 minutes after the last one.
  `OLLAMA_KEEP_ALIVE` applies only to calls that set no `keep_alive`, such as the warm-up above.
* Ollama is a separate service with its own network access, outside Praktika's allow-list.
  Firewall it.

To use an OpenAI-compatible server (for example vLLM serving the drafting model) instead, set
`PRAKTIKA_LLM_PROVIDER=openai_compat`, `PRAKTIKA_LLM_BASE_URL` to its base URL (host
allow-listed) and `PRAKTIKA_LLM_MODEL` and `PRAKTIKA_LLM_FALLBACK_MODEL` both to the served model
name. Praktika then asks for JSON-schema guided decoding. A bearer token, if the server needs one,
is read from `PRAKTIKA_LLM_TOKEN` in the process environment. This client has been tested against a
mocked server only.

## 9. Retention timer

Audio, transcripts and drafts are deleted by timers, which are checked at every retention run: at
the start of every command that opens the meeting store (except `audit verify`), once when
`praktika serve` starts (not again while it runs), with `praktika retention run`, and hourly from a
systemd timer. Without the timer, nothing is deleted until someone runs one of those commands, so
the timer is what makes the deadlines hold. The deadlines themselves (audio after approval or
discard and at most 24 hours after conversion for Internal meetings; transcripts 14 days after
approval and at most 60 days, a maximum fixed in code; unapproved drafts 30 days after approval
or after the latest draft) are
described in the README's [Before real use](../README.md#before-real-use). Install the timer as a
system unit that runs as the service account:

```bash
# [root]
PRAKTIKA_ENV_FILE=/etc/praktika/praktika.env /opt/praktika/.venv/bin/praktika \
    retention install --unit-dir /etc/systemd/system --run-as praktika
systemctl daemon-reload && systemctl enable --now praktika-retention.timer
systemctl list-timers praktika-retention.timer
```

The installer pins `PRAKTIKA_ENV_FILE`, `PRAKTIKA_DATA_DIR` and the virtual environment's
interpreter into `praktika-retention.service` (a oneshot with `UMask=0077` and
`NoNewPrivileges=yes`) and writes `praktika-retention.timer` (`OnCalendar=hourly`,
`Persistent=true`). It refuses to write a unit that would run as root. Output goes to the journal
(`journalctl -u praktika-retention.service`). The timer does not need the vault key. Without
`--unit-dir`, the installer writes a user unit for the installing account instead, which runs only
while that account has a session unless lingering is enabled (`loginctl enable-linger`).

## 10. `praktika doctor`

```bash
sudo -u praktika -E /opt/praktika/.venv/bin/praktika doctor      # --json for a machine-readable copy
```

Thirteen checks; only a `FAIL` sets exit code 1. On a fresh host, expect exit 0 with about four
warnings:

| Check | Expected | Meaning |
|---|---|---|
| `env_file` | OK, names `/etc/praktika/praktika.env` | WARN `none loaded`, or another file, means `PRAKTIKA_ENV_FILE` did not reach the process. WARN on the right file means it sets no `PRAKTIKA_` variable yet |
| `ffmpeg` | OK | WARN blocks audio ingest only |
| `ollama` | OK, model present | FAIL: Ollama unreachable or the model is missing. WARN: the model's stored `num_ctx` is lower than configured, which is harmless because Praktika sends it per call |
| `models` | OK, `verified: stt_en` | WARN until a register exists. With the speech backend on HTTP a mismatch is a WARN here; `praktika models verify` is the check that exits 1 |
| `disk_encryption` | OK or WARN | OK only when the data directory is on a dm-crypt mapping. Self-encrypting drives, storage-array encryption and LVM on LUKS are invisible to this check and show as WARN; confirm encryption another way |
| `data_dir` | OK, mode 700 | FAIL on any other mode |
| `no_egress` | OK, `allowed_hosts=localhost, 127.0.0.1` | FAIL: a configured URL is outside the allow-list |
| `memory` | OK | Host memory, not GPU memory |
| `identity` | WARN, `source=local` | Expected without single sign-on. Record the actor string it shows |
| `audit_chain` | WARN before the first ingest, OK after | If it does not turn OK after the first ingest, stop and investigate |
| `prompts` | OK, matches the pin | WARN: the prompt files differ from the pinned hash; review them before drafting |
| `vault_key` | OK | FAIL: `PRAKTIKA_VAULT_KEY` is unset or not a valid Fernet key |
| `endpoint_agent` | WARN | Looks only for a marker file (`<data_dir>/endpoint-agent.ok`) that a log-shipping agent is expected to write |

## 11. First runs, with synthetic data

Run these from the checkout, because the fixture paths are relative, and as the service account.
Every fixture is synthetic.

```bash
cd /opt/praktika
p() { sudo -u praktika -E /opt/praktika/.venv/bin/praktika "$@"; }

# a Teams transcript: no speech model needed
p ingest tests/fixtures/synthetic_en.vtt --title "Install check: VTT" \
    --type general --class internal \
    --notified --no-objections --method chat --teams-transcription-started \
    --purpose "Installation check with synthetic data" --ack-all-scope
# the same with the Teams recap document, tests/fixtures/synthetic_recap.docx,
# and the management-committee template, --type mancom; add --roster <file.yaml> for an
# attendee list (the synthetic roster in tests/fixtures/ shows the format)

p eval --llm fake                     # six synthetic golden meetings; must print Gate: PASS
p audit verify                        # hash chain, head file and database agree
p models verify
p retention run --dry-run
p export <meeting-id> --format docx   # must be refused: minutes are not approved
```

For audio, ingest the synthetic recording `tests/fixtures/synthetic_meeting.wav` (or any recording
of your own synthetic speech) with `--platform in_room --method spoken
--no-teams-transcription-started`. If
the speech server is not running, the ingest fails with `STT request failed`; see section 16.

Each ingest prints `Draft vN ready for M-YYYYMMDD-xxxx: ...` and a review URL. Review the draft
(section 12) **before** approving: approval closes the meeting, and the next retention run (the
next command that opens the meeting store, or the hourly timer) deletes its audio, after which
click-to-play returns 404. Then:

```bash
p approve <meeting-id> --reason accurate
p export <meeting-id> --format docx   # written 0600 under /var/lib/praktika/exports
```

## 12. Review page

### Local mode, over SSH

On the server:

```bash
sudo -u praktika -E /opt/praktika/.venv/bin/praktika serve
```

It prints `Review UI on http://127.0.0.1:8793/?t=<token>`. On the reviewer's own machine:

```bash
ssh -N -L 8793:127.0.0.1:8793 <operator>@<server>
```

Then open the URL exactly as printed, token included.

In local mode `serve` binds `127.0.0.1` only and refuses any other address. The token is read once
from the address, kept in the tab's session storage and sent as a header thereafter; opening the
page without it gives 401 on every call. The token is persistent per data directory and never
expires; to issue a new one, stop `serve` and delete `/var/lib/praktika/review.token`.

The page acts as the account that started `serve` and lists the meetings that account organises.
A meeting ingested with `--organiser <someone-else>` is visible only to that organiser. There is no
systemd unit for `serve`: it does not survive a reboot.

### Service mode, behind a gateway

For a shared deployment, put `praktika serve` behind a gateway that terminates TLS and performs the
OIDC sign-in, and set:

```ini
PRAKTIKA_MODE=service
PRAKTIKA_IDENTITY_PROVIDER=oidc
PRAKTIKA_OIDC_ISSUER=https://<identity-provider>/<tenant>/v2.0
PRAKTIKA_OIDC_AUDIENCE=api://praktika
PRAKTIKA_OIDC_JWKS_URL=https://<identity-provider>/<tenant>/discovery/v2.0/keys
PRAKTIKA_REVIEW_HOST=<internal interface address>
PRAKTIKA_ALLOWED_HOSTS=["praktika.<your-domain>","<identity-provider>","localhost","127.0.0.1"]
```

* Service mode refuses to start unless the identity provider is `oidc`, whatever the bind address.
* Tokens must be RS256 JWTs whose signature verifies against the JWKS and whose issuer, audience
  and expiry are valid. Roles come from the `groups` claim (group **names**, not object IDs) or the
  `roles` claim: `Praktika-Users` act on their own meetings, `Praktika-Secretaries` on any
  non-private meeting, `Praktika-DPO` and `Praktika-Admins` read only. No other header is trusted.
* The static review page sends no `Authorization` header: the gateway must inject the bearer
  token.
* `PRAKTIKA_ALLOWED_HOSTS` is both the egress allow-list and, in service mode, the list of trusted
  `Host` headers, so the page's own host name must be in it, and anything you add becomes a
  trusted origin too. Use fully qualified names, no wildcards.
* The command line has no identity under OIDC: run ingests with
  `PRAKTIKA_IDENTITY_PROVIDER=session` in their own environment, and name the real organiser with
  `--organiser <upn>`.
* With `PRAKTIKA_AUDIT_SINK=stdout`, a copy of every audit line is written to
  `/var/lib/praktika/audit-forward.jsonl` (0600, rotated at 64 MiB with five backups) for your log
  shipper to tail; nothing goes to standard output despite the name.

This path is implemented and tested with synthetic tokens only.

## 13. Day-to-day operation

* **Getting transcripts onto the server.** Use one inbox directory owned by `praktika` and
  writable by its group, so that the service account can read what operators copy in:

  ```bash
  # [root]
  install -d -m 2770 -o praktika -g praktika /srv/praktika-inbox
  ```

  Copy the transcript in (for example with `scp`), `chmod 0640` it, ingest it, and delete it once
  the ingest has printed its draft line. `praktika ingest` never deletes the file it is given. Keep
  the inbox out of backups.
* **One run per meeting at a time.** `start`, `ingest`, `transcribe`, `generate` and a regenerate
  from the review page each take the meeting's run lock. A second run on the same meeting is
  refused with an error naming the command that holds it; review-page edits on that meeting return
  409 until it finishes. A lock left by a process that no longer exists on this host is taken over,
  and the takeover is audited.
* **`praktika abort <meeting-id>` always wins.** It discards the meeting even while a run holds the
  lock, overwrites any audio left in its folder, and the run stops at its next step, removing only
  what it stored itself. It refuses approved, purged and held meetings.
* **Legal hold**: `praktika hold set <meeting-id> --reason "<why>"` blocks every retention timer for
  that meeting until `praktika hold clear`.
* **Data-subject requests**: `praktika dsar find|export|delete --participant "<name or UPN>"`.

## 14. Audit events

Every event is one JSON line in `/var/lib/praktika/audit.jsonl` with the fields `ts`, `actor`,
`actor_source`, `event`, `meeting_id`, `classification`, `object`, `detail`, `model`, `prompt_sha`,
`prev_hash` and `hash`; each line carries the hash of the line before it. Model calls are recorded
with sizes, timings and the prompt hash, never the content. This is the complete vocabulary (the
`audit.append` call sites in `src/`), which a log-shipping parser should be written from:

| Area | Events |
|---|---|
| Consent and scope | `consent.recorded`, `scope.refused`, `meeting.organiser_named` |
| Capture and ingest | `capture.started`, `capture.stopped`, `capture.aborted`, `ingest.vtt`, `ingest.docx`, `ingest.file`, `ingest.failed`, `stt.completed`, `diarize.completed`, `redact.applied`, `run.lock_taken_over` |
| Drafting | `llm.call`, `minutes.drafted` |
| Review | `review.opened`, `review.item`, `review.speaker_mapped`, `review.approved`, `review.discarded`, `review.reopened`, `audio.read`, `auth.denied` |
| Export and records | `export.written`, `dsar.export`, `dsar.delete`, `hold.set`, `hold.released` |
| Retention and models | `retention.deleted`, `retention.failed`, `models.verified` |

Points to know when writing detections:

* `ingest.failed` carries `stage` (`convert`, `transcribe`, `diarize`, `assemble`, `register`,
  `redact`, `draft` or `capture`), `error` (the exception's type name only, never its message) and
  `purged` (the audio files that run removed). `error=RunStoppedError` means an abort, a discard on
  the review page or a DSAR deletion moved the meeting on while the run was working; `stopped_by`
  says which.
* The verifier writes no event: an item it removes for citing an instruction-like segment is
  visible only as a flag in the stored minutes.
* `models.verified` is not written at ingest when the speech backend is HTTP, because nothing is
  loaded locally; run `praktika models verify` on a schedule instead.
* Some `detail` fields carry text a user typed: the reviewer's reason on `review.item`,
  `review.approved` and `review.discarded`, speaker labels on `review.speaker_mapped`, and the
  request route on `auth.denied`. They are JSON-encoded, so they cannot break a line, but treat them
  as untrusted.
* Refusals by the review server's request guard (wrong `Host`, missing CSRF header or session token)
  are logged to the application log, not to the audit chain.

## 15. Backups

A backup of the data directory outlives every retention timer. Either exclude
`/var/lib/praktika` (and the inbox) from backup, or make the backup cover the database, the
exports, `audit.jsonl` **and** `audit.jsonl.head` together, inherit the retention clock, and honour
legal holds. After a restore, run `praktika audit verify`, `praktika models verify` and
`praktika doctor`. The checkout, the env file and the weights directory hold no meeting content.

## 16. Failure modes

| Symptom | Likely cause and fix |
|---|---|
| Every command exits 1 at once with an error telling you to export `PRAKTIKA_ENV_FILE` | The variable is not exported in this shell, and `sudo -E` kept your own `HOME`. Export it and run again |
| Every command refuses at once with an error about the environment file | `PRAKTIKA_ENV_FILE` names a file that does not exist or that `praktika` cannot read (it should be 0640, group `praktika`) |
| Settings are all defaults | `sudo` ran without `-E`, or the variable was not exported; `doctor`'s `env_file` line names the file actually loaded |
| `error: cannot use the data directory ...` | The directory belongs to another account, or the command is not running as `praktika` |
| Start-up exits 1 with "configured path(s) not found" | The checkout was moved or deleted, or Praktika was installed without `-e`; the prompts and glossary are read from the checkout |
| `EgressError` at start-up | A configured URL's host is not in `PRAKTIKA_ALLOWED_HOSTS`. This is the control working |
| `STT server returned 404` on every chunk | `--served-model-name` is not exactly `whisper-large-v3`, or `PRAKTIKA_STT_HTTP_URL` carries a path |
| `STT request failed: ... Connection refused` | The speech server is not running or not on `127.0.0.1:8801` (`docker ps`, `docker logs praktika-stt-en`). The failed run has purged its audio and left the meeting at `created`; start the server, ingest again, and remove the old meeting with `praktika abort` |
| `the transcript has no segments` | The recording had no speech the server could transcribe, or the transcript file is empty. A registered WAV stays under its timer: fix the cause, then `praktika transcribe <id>` and `praktika generate <id>`. Otherwise `praktika abort <id>` |
| An ingest says the transcript is stored, then fails | Drafting failed (Ollama down or the model missing). Fix it and run `praktika generate <id>`; do not ingest again |
| Ingest fails at redaction naming `PRAKTIKA_VAULT_KEY` | The key is not exported. Export it; retry audio with `transcribe` and `generate`, or re-ingest a transcript file and abort the old meeting |
| A run is refused: "a `<command>` run is working on this meeting" | Another run holds the lock. Wait for it. If its process is gone it is taken over automatically |
| A meeting stays at `transcribing` or `drafting` after its run disappeared | The run was killed. With a transcript stored, `praktika generate <id>` takes over; without one, `praktika abort <id>` |
| Speech runs at about real time and the GPU is idle | The `cuda` extra was installed and faster-whisper is running on the CPU. Recreate the virtual environment without extras and keep `PRAKTIKA_STT_EN=http` |
| Citations are always up to 28 s wide; click-to-play lands vaguely | Word timestamps are not coming back; run the probe in section 7 |
| Minutes are thin, long meetings worse than short ones | Context truncation: confirm `PRAKTIKA_LLM_NUM_CTX=32768` |
| A long meeting fails with an HTTP error naming an unknown model | The fallback model is not served or not pulled; set `PRAKTIKA_LLM_FALLBACK_MODEL` to the main model |
| A long meeting fails only on "regenerate" | Regenerate sends the whole transcript in one call; see [LIMITATIONS.md](LIMITATIONS.md) |
| The speech container exits at start with an out-of-memory error, or "less than desired GPU memory utilization" | Something else holds GPU memory (Ollama with `OLLAMA_NUM_PARALLEL` above 1, a bigger model, another container). `ollama stop qwen2.5:14b`, start the container, then load the model again |
| The speech container exits with "Error in memory profiling" or "No available memory for the cache blocks" | Another process took or freed memory while vLLM measured it, usually Ollama loading or unloading. Stop the model, restart the container, wait for `/v1/models` |
| `ollama ps` shows a CPU share, and drafting is several times slower | Too little GPU memory was free when the model loaded. `100% CPU` means Ollama sees no GPU: `lib/ollama` missing, or a wrong `CUDA_VISIBLE_DEVICES` |
| `exec format error` when the container starts | The image is for the other CPU architecture |
| The host becomes sluggish or a container is killed (unified memory) | `--gpu-memory-utilization` is too high, or something else is running on the host |
| `serve` refuses to start: "review_host is not loopback" | `PRAKTIKA_REVIEW_HOST` was changed in local mode. Use the SSH tunnel; a network bind needs service mode with OIDC |
| Every API call from the browser returns 401 | The page was opened without the `?t=` token; open exactly the printed URL |
| Audio playback returns 404 after a review | The meeting was approved or discarded, and the retention sweep deleted its audio |
| `praktika audit verify` fails with "audit.jsonl is missing but events were recorded" | Someone deleted or moved the audit log. Treat it as an incident |
| An action fails with "audit log lock ... not acquired within 30 s" | Another Praktika process hangs while holding the lock: `fuser /var/lib/praktika/audit.jsonl.lock`. Do not delete the lock file while processes run |
| `retention install` refuses: "runs as root" | Pass `--unit-dir /etc/systemd/system --run-as praktika` |
| Retention never runs unattended | The timer is not enabled (`systemctl list-timers praktika-retention.timer`), or it is a user unit without lingering |
| The review page does not come back after a reboot | There is no unit for `serve`; start it again, and export the vault key again in that shell |
