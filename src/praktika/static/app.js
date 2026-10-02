/* Praktika review UI: vanilla JS, no build step. Talks to /api only. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const state = { meetingId: null, detail: null, run: null };
const ARABIC = /[\u0600-\u06FF]/;

/* Local mode: `praktika serve` prints the review URL with the data directory's persistent
   session token (?t=...). It is kept in sessionStorage, stripped from the address bar and sent
   as a header on every API call. */
function sessionToken() {
  const params = new URLSearchParams(window.location.search);
  const given = params.get("t");
  if (given) {
    try { sessionStorage.setItem("praktika.token", given); } catch (_) { /* private mode */ }
    params.delete("t");
    const rest = params.toString();
    history.replaceState(null, "", window.location.pathname + (rest ? `?${rest}` : "") + window.location.hash);
    return given;
  }
  try { return sessionStorage.getItem("praktika.token") || ""; } catch (_) { return ""; }
}
const TOKEN = sessionToken();

function apiHeaders(extra = {}) {
  const h = { "X-Praktika-Review": "1", ...extra };
  if (TOKEN) h["X-Praktika-Token"] = TOKEN;
  return h;
}

const SECTION_LABELS = {
  decisions: "Decisions",
  actions: "Actions",
  open_questions: "Open questions",
  risks: "Risks",
};
const FIELD_OF = {
  decisions: "statement",
  actions: "description",
  open_questions: "question",
  risks: "description",
};

async function api(path, options = {}) {
  const init = { ...options, headers: apiHeaders({ "Content-Type": "application/json" }) };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  const resp = await fetch(path, init);
  if (!resp.ok) {
    let msg = `${resp.status} ${resp.statusText}`;
    try { msg = (await resp.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return resp.status === 204 ? null : resp.json();
}

function notice(text, kind = "info") {
  const el = $("#notice");
  el.textContent = text;
  el.className = `notice ${kind}`;
  el.classList.toggle("hidden", !text);
}

function hms(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  return `${h}:${m}:${String(s % 60).padStart(2, "0")}`;
}

function fillReasons(select, codes) {
  select.innerHTML = "";
  for (const code of codes) {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = code.replace(/_/g, " ");
    select.appendChild(opt);
  }
}

/* ---------------------------------------------------------------- meeting list */

async function showList() {
  state.meetingId = null;
  stopRunWatch();
  $("#view-review").classList.add("hidden");
  $("#view-list").classList.remove("hidden");
  $("#crumb-meeting").classList.add("hidden");
  const rows = await api("/api/meetings");
  const body = $("#meeting-table tbody");
  body.innerHTML = "";
  $("#list-empty").classList.toggle("hidden", rows.length > 0);
  rows.sort((a, b) => (a.review_status === "approved") - (b.review_status === "approved"));
  for (const m of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><a href="#/meetings/${m.id}" dir="auto"></a><div class="muted small">${m.id}</div></td>
      <td></td><td><span class="pill"></span></td><td><span class="pill state"></span></td>
      <td class="num"></td><td class="muted"></td>`;
    tr.querySelector("a").textContent = m.title;
    tr.children[1].textContent = m.meeting_type.replace(/_/g, " ");
    tr.querySelector(".pill").textContent = m.classification;
    tr.querySelector(".state").textContent = m.state.replace(/_/g, " ");
    tr.querySelector(".state").dataset.state = m.state;
    tr.children[4].textContent = m.blocking ? String(m.blocking) : "—";
    tr.children[5].textContent = new Date(m.started_at).toLocaleString("en-GB");
    body.appendChild(tr);
  }
}

/* ---------------------------------------------------------------- review view */

/* While a run (a generate from the command line, a regenerate from another tab) is working on
   the meeting, the server refuses edits with 409; the page greys them out instead, says which
   run is working, and reloads the meeting once the run has finished. */
const RUN_POLL_MS = 5000;
let runTimer = null;

async function runStatus(meetingId) {
  try {
    return await api(`/api/meetings/${encodeURIComponent(meetingId)}/run`);
  } catch (_) {
    return { running: false };
  }
}

function stopRunWatch() {
  if (runTimer) clearTimeout(runTimer);
  runTimer = null;
}

function watchRun(meetingId) {
  stopRunWatch();
  if (!state.run || !state.run.running) return;
  runTimer = setTimeout(async () => {
    runTimer = null;
    if (state.meetingId !== meetingId) return;
    const run = await runStatus(meetingId);
    if (state.meetingId !== meetingId) return;
    if (run.running) {
      state.run = run;
      watchRun(meetingId);
      return;
    }
    try {
      await showReview(meetingId); // the run has finished: show what it left
    } catch (err) {
      notice(err.message, "error");
    }
  }, RUN_POLL_MS);
}

async function showReview(meetingId) {
  state.meetingId = meetingId;
  state.detail = await api(`/api/meetings/${encodeURIComponent(meetingId)}`);
  state.run = await runStatus(meetingId);
  $("#view-list").classList.add("hidden");
  $("#view-review").classList.remove("hidden");
  $("#crumb-meeting").textContent = state.detail.meeting.title;
  $("#crumb-meeting").classList.remove("hidden");
  notice("");
  render();
  watchRun(meetingId);
}

function render() {
  const d = state.detail;
  const m = d.meeting;
  const mins = d.minutes;
  $("#r-title").textContent = m.title;
  $("#r-meta").textContent = `${m.id} · ${m.meeting_type.replace(/_/g, " ")} · ${m.classification} · ${m.state.replace(/_/g, " ")} · organiser ${m.organiser}`;
  fillReasons($("#reason-select"), d.reason_codes);
  const closed = ["approved", "discarded", "purged"].includes(m.state);
  const busy = Boolean(state.run && state.run.running);
  const frozen = closed || busy; // no edits while the minutes are closed or a run works on them
  $("#btn-approve").disabled = frozen || !mins;
  $("#btn-discard").disabled = closed || !mins; // a discard, like an abort, stops a run
  renderFlags(d.flags, frozen);
  renderMinutes(mins, frozen);
  renderSpeakers(d.unmapped_speakers, m.roster, frozen);
  renderTranscript(d.segments);
  renderProvenance(mins);
  const regen = $("#regen-section");
  regen.innerHTML = "";
  for (const s of d.sections) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s.replace(/_/g, " ");
    regen.appendChild(opt);
  }
  $("#btn-regen").disabled = frozen || !d.regenerate_available;
  $("#btn-regen").title = d.regenerate_available ? "" : "No LLM configured for this session";
  if (busy) {
    const what = state.run.message || "a run is working on this meeting";
    notice(`${what.charAt(0).toUpperCase()}${what.slice(1)}; editing is paused until it finishes.`, "warn");
  }
}

function renderFlags(flags, closed) {
  const list = $("#flag-list");
  list.innerHTML = "";
  const open = flags.filter((f) => !f.cleared);
  const blocking = open.filter((f) => f.priority === 1).length;
  $("#flag-count").textContent = open.length ? `${open.length} open · ${blocking} blocking` : "none open";
  for (const f of flags) {
    const li = document.createElement("li");
    li.className = `flag p${f.priority}${f.cleared ? " cleared" : ""}`;
    const kind = document.createElement("span");
    kind.className = "flag-kind";
    kind.textContent = `P${f.priority} · ${f.kind.replace(/_/g, " ")}`;
    const detail = document.createElement("p");
    detail.className = "flag-detail";
    detail.dir = "auto";
    detail.textContent = f.detail;
    li.append(kind, detail);
    if (f.refs && f.refs.length) li.appendChild(citeChips(f.refs));
    if (f.cleared) {
      const by = document.createElement("span");
      by.className = "muted small";
      by.textContent = `cleared by ${f.cleared_by}`;
      li.appendChild(by);
    } else if (!closed) {
      const row = document.createElement("div");
      row.className = "flag-actions";
      if (f.kind === "uncited_item_removed" && f.item_json) {
        const item = JSON.parse(f.item_json);
        const restore = document.createElement("button");
        restore.textContent = `Restore ${item.id}`;
        restore.onclick = () => reviewItem(item.id, "restore", "cited_elsewhere");
        row.appendChild(restore);
      }
      const clear = document.createElement("button");
      clear.textContent = "Mark resolved";
      clear.className = "quiet";
      clear.onclick = () => post(`/api/minutes/${state.meetingId}/flags/${f.n}/clear`, {});
      row.appendChild(clear);
      li.appendChild(row);
    }
    list.appendChild(li);
  }
}

function renderMinutes(mins, closed) {
  const sections = $("#sections");
  sections.innerHTML = "";
  $("#summary").value = mins ? mins.summary : "";
  $("#minutes-version").textContent = mins ? `v${mins.version} · ${mins.review.status.replace(/_/g, " ")}` : "";
  if (!mins) {
    sections.innerHTML = '<p class="muted">No minutes have been drafted for this meeting yet.</p>';
    return;
  }
  for (const [name, label] of Object.entries(SECTION_LABELS)) {
    const h = document.createElement("h3");
    h.textContent = `${label} (${mins[name].length})`;
    sections.appendChild(h);
    const ul = document.createElement("ul");
    ul.className = "items";
    for (const item of mins[name]) ul.appendChild(itemRow(name, item, closed));
    sections.appendChild(ul);
  }
  if (mins.follow_ups && mins.follow_ups.length) {
    const h = document.createElement("h3");
    h.textContent = "Follow-ups";
    const ul = document.createElement("ul");
    for (const f of mins.follow_ups) {
      const li = document.createElement("li");
      li.dir = "auto";
      li.textContent = f;
      ul.appendChild(li);
    }
    sections.append(h, ul);
  }
}

/* A resolved due date is shown with the phrase actually spoken beside it, so a reviewer can
   catch a date resolved wrongly ("November the 2nd" read as 2 October); an unresolved phrase
   is shown as spoken. Exports keep their own format. */
function dueLabel(dueDate, dueText) {
  const spoken = (dueText || "").trim();
  if (!dueDate) return spoken;
  return spoken && spoken !== dueDate ? `${dueDate} (“${spoken}”)` : dueDate;
}

function itemRow(name, item, closed) {
  const node = $("#tpl-item").content.firstElementChild.cloneNode(true);
  const field = FIELD_OF[name];
  node.dataset.id = item.id;
  $(".item-id", node).textContent = item.id;
  const extra = [];
  if (item.kind) extra.push(item.kind.replace(/_/g, " "));
  if (item.decided_by) extra.push(`by ${item.decided_by}`);
  if (item.owner) extra.push(`owner ${item.owner} (${item.owner_confidence})`);
  const due = dueLabel(item.due_date, item.due_text);
  if (due) extra.push(`due ${due}`);
  if (item.severity) extra.push(`${item.severity} severity`);
  $(".item-extra", node).dir = "auto";
  $(".item-extra", node).textContent = extra.join(" · ");
  const text = $(".item-text", node);
  text.value = item[field];
  text.readOnly = closed;
  $(".cites", node).appendChild(citeChips(item.refs || []));
  fillReasons($(".item-reason", node), state.detail.reason_codes);
  const reason = () => $(".item-reason", node).value;
  $(".accept", node).onclick = () => reviewItem(item.id, "accept", reason());
  $(".save", node).onclick = () => reviewItem(item.id, "modify", reason(), item[field], text.value);
  $(".reject", node).onclick = () => reviewItem(item.id, "reject", reason(), item[field]);
  for (const b of node.querySelectorAll("button")) b.disabled = closed;
  return node;
}

function citeChips(refs) {
  const wrap = document.createElement("span");
  wrap.className = "chips";
  for (const r of refs) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = `${r.segment_id} ${hms(r.start_s)}`;
    chip.title = r.quote;
    chip.onclick = () => jumpTo(r.segment_id, r.start_s, r.end_s);
    wrap.appendChild(chip);
    if (ARABIC.test(r.quote)) {
      // Arabic evidence is shown inline (a tooltip is not readable right-to-left).
      const quote = document.createElement("span");
      quote.className = "chip-quote";
      quote.dir = "auto";
      quote.textContent = `“${r.quote}”`;
      wrap.appendChild(quote);
    }
  }
  return wrap;
}

function renderSpeakers(labels, roster, closed) {
  const box = $("#speaker-map");
  box.innerHTML = "";
  if (!labels.length) return;
  const h = document.createElement("h3");
  h.textContent = "Name the speakers";
  box.appendChild(h);
  for (const label of labels) {
    const row = document.createElement("div");
    row.className = "row";
    const l = document.createElement("label");
    l.textContent = label;
    const sel = document.createElement("select");
    sel.dataset.label = label;
    sel.disabled = closed;
    sel.innerHTML = '<option value="">— leave as is —</option>';
    for (const a of roster) {
      const opt = document.createElement("option");
      opt.value = a.name;
      opt.textContent = a.name;
      sel.appendChild(opt);
    }
    row.append(l, sel);
    box.appendChild(row);
  }
  const btn = document.createElement("button");
  btn.textContent = "Apply names";
  btn.disabled = closed;
  btn.onclick = () => {
    const mapping = {};
    for (const s of box.querySelectorAll("select")) if (s.value) mapping[s.dataset.label] = s.value;
    post(`/api/meetings/${state.meetingId}/speakers`, mapping);
  };
  box.appendChild(btn);
}

function renderTranscript(segments) {
  const ol = $("#segments");
  ol.innerHTML = "";
  for (const s of segments) {
    const li = document.createElement("li");
    li.id = `seg-${s.id}`;
    li.className = `seg lang-${s.language}`;
    li.innerHTML = '<span class="seg-time"></span><span class="seg-speaker"></span><span class="seg-text" dir="auto"></span>';
    $(".seg-time", li).textContent = hms(s.start);
    $(".seg-speaker", li).textContent = s.speaker;
    $(".seg-text", li).textContent = s.text;
    li.onclick = () => play(s.start, s.end, s.track);
    ol.appendChild(li);
  }
}

function renderProvenance(mins) {
  const p = $("#provenance");
  if (!mins) { p.textContent = ""; return; }
  const v = mins.provenance;
  p.textContent = `Generated by ${v.generator_model} (${v.model_digest}) · prompts ${v.prompt_version} ${v.prompt_sha256.slice(0, 12)} · glossary ${v.glossary_sha256.slice(0, 12)} · transcript ${v.transcript_sha256.slice(0, 12)} · git ${v.git_sha}${v.degraded ? " · DEGRADED MODEL" : ""} · ${v.generated_at}`;
}

/* ---------------------------------------------------------------- interactions */

async function jumpTo(segmentId, start, end) {
  for (const el of document.querySelectorAll(".seg.active")) el.classList.remove("active");
  const el = document.getElementById(`seg-${segmentId}`);
  if (el) {
    el.classList.add("active");
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  const seg = (state.detail.segments || []).find((s) => s.id === segmentId);
  await play(start, end, seg ? seg.track : null);
}

async function play(start, end, track = null) {
  if (!state.detail.audio_available) return;
  let q = `start=${encodeURIComponent(start)}&end=${encodeURIComponent(Math.max(end, start + 0.5))}`;
  if (track) q += `&track=${encodeURIComponent(track)}`;
  try {
    const resp = await fetch(`/api/meetings/${encodeURIComponent(state.meetingId)}/audio?${q}`, { headers: apiHeaders() });
    if (!resp.ok) throw new Error(`audio ${resp.status}`);
    const blob = await resp.blob();
    const player = $("#player");
    if (player.dataset.url) URL.revokeObjectURL(player.dataset.url);
    player.dataset.url = URL.createObjectURL(blob);
    player.src = player.dataset.url;
    await player.play();
  } catch (err) {
    notice(`Audio unavailable: ${err.message}`, "warn");
  }
}

/* Exports are fetched rather than linked: a plain <a download> cannot send the session token or
   the review header, so the server answers 401. The response is saved through a Blob URL under
   the name the server gives in Content-Disposition. */
function dispositionName(header, fallback) {
  const safe = (name) => name.replace(/[\\/]/g, "_");
  const star = /filename\*\s*=\s*(?:UTF-8'')?([^;]+)/i.exec(header || "");
  if (star) {
    try {
      const decoded = safe(decodeURIComponent(star[1].trim().replace(/^"|"$/g, "")));
      if (decoded) return decoded;
    } catch (_) { /* fall through */ }
  }
  const plain = /filename\s*=\s*("([^"]*)"|[^;]+)/i.exec(header || "");
  const name = plain ? (plain[2] !== undefined ? plain[2] : plain[1].trim()) : "";
  return safe(name) || fallback;
}

async function downloadExport(fmt) {
  if (!state.meetingId) return;
  const fallback = `${state.meetingId}.${fmt}`;
  try {
    const resp = await fetch(`/api/export/${encodeURIComponent(state.meetingId)}.${fmt}`, { headers: apiHeaders() });
    if (!resp.ok) {
      let msg = `${resp.status} ${resp.statusText}`;
      try { msg = (await resp.json()).detail || msg; } catch (_) { /* not JSON */ }
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = dispositionName(resp.headers.get("Content-Disposition"), fallback);
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notice(`Exported ${a.download}.`, "info");
  } catch (err) {
    notice(`Export failed: ${err.message}`, "error");
  }
}

async function post(path, body) {
  try {
    await api(path, { method: "POST", body });
    await showReview(state.meetingId);
  } catch (err) {
    notice(err.message, "error");
    const meetingId = state.meetingId;
    const run = meetingId ? await runStatus(meetingId) : { running: false };
    if (run.running && meetingId === state.meetingId && state.detail) {
      state.run = run; // refused because a run has started meanwhile: pause the edits
      render();
      watchRun(meetingId);
    }
  }
}

function reviewItem(itemId, action, reasonCode, before = null, after = null) {
  return post(`/api/minutes/${state.meetingId}/items/${encodeURIComponent(itemId)}`, {
    action, reason_code: reasonCode, before, after,
  });
}

$("#btn-approve").onclick = () => {
  if (!window.confirm("Approve these minutes as the named reviewer? Audio will be scheduled for deletion.")) return;
  post(`/api/minutes/${state.meetingId}/approve`, { reason_code: $("#reason-select").value });
};
$("#btn-discard").onclick = () => {
  if (!window.confirm("Discard this draft? The transcript and draft will be deleted under the retention schedule.")) return;
  post(`/api/minutes/${state.meetingId}/discard`, { reason_code: $("#reason-select").value });
};
$("#btn-regen").onclick = () => {
  const instruction = $("#regen-instruction").value.trim();
  if (instruction.length < 3) { notice("Give the re-draft a short instruction.", "warn"); return; }
  notice("Regenerating — this can take a minute or two.", "info");
  post(`/api/minutes/${state.meetingId}/regenerate`, { section: $("#regen-section").value, instruction });
};
$("#link-export-md").onclick = (ev) => { ev.preventDefault(); downloadExport("md"); };
$("#link-export-docx").onclick = (ev) => { ev.preventDefault(); downloadExport("docx"); };
$("#nav-list").onclick = (ev) => { ev.preventDefault(); window.location.hash = ""; };

async function route() {
  // Routes: "#/meetings/<id>" (as printed by the CLI) and the short form "#<id>".
  const id = window.location.hash.replace(/^#\/?(meetings\/)?/, "");
  try {
    if (id) await showReview(id); else await showList();
  } catch (err) {
    notice(err.message, "error");
    if (id) { $("#view-review").classList.remove("hidden"); $("#view-list").classList.add("hidden"); }
  }
}

api("/api/me").then((me) => { $("#viewer").textContent = `${me.display} · ${me.source}`; }).catch(() => {});
window.addEventListener("hashchange", route);
route();
