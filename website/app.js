// Sends each dropped file to the server in turn and adds one row per file to both versions.
const VERSIONS = ["gru", "transformer"];
const drop = document.getElementById("drop"), input = document.getElementById("files");
const statusLine = document.getElementById("status"), clearButton = document.getElementById("clear");
const orderSwitch = document.getElementById("order"), orderNote = document.getElementById("order-note");
const NOTES = {
  "123": "Order of the brief: the classifier sees the file as received, and only photographs are denoised.",
  "213": "Every file is denoised first, then classified: more noisy photographs are kept, at the cost of a few more non-photos accepted.",
};
const queue = [];
let order = "123";
let busy = false, counts = { total: 0, done: 0, photos: 0, rejected: 0, errors: 0 };

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

function step(label, title, body) {
  const s = el("div", "step");
  s.append(el("p", "step-label", label), el("p", "step-title", title));
  if (body) s.append(body);
  return s;
}

function figure(src, alt, r) {
  const img = el("img"); img.src = src; img.alt = alt; img.loading = "lazy";
  img.title = "Click to compare at full size";
  img.addEventListener("click", () => compare(r));
  return img;
}

// Full-size view of the input next to the denoised image, where the removed noise is visible.
const viewer = document.getElementById("viewer");
function compare(r) {
  const panes = [["Input", r.input_full], ["Denoised", r.denoised_full]].filter(p => p[1]);
  const grid = el("div", "viewer-grid");
  for (const [label, src] of panes) {
    const f = el("figure"); const img = el("img"); img.src = src; img.alt = label + " " + r.name;
    f.append(el("figcaption", null, label), img); grid.append(f);
  }
  viewer.replaceChildren(el("p", "viewer-title", r.name + " · click anywhere to close"), grid);
  viewer.showModal();
}
viewer.addEventListener("click", () => viewer.close());

function effect(r) {
  const e = r.effect;
  if (e.mean_change < 0.5) return el("p", "effect", "Unchanged: no noise to remove (mean change " + e.mean_change.toFixed(1) + " / 255)");
  const drop = 100 * (1 - e.grain_after / Math.max(e.grain_before, 1e-6));
  return el("p", "effect", "Mean change " + e.mean_change.toFixed(1) + " / 255 · fine grain " +
            e.grain_before.toFixed(1) + " → " + e.grain_after.toFixed(1) + " (" + (drop >= 0 ? "−" : "+") + Math.abs(drop).toFixed(0) + " %)");
}

function denoisedStep(r) {
  const box = el("div");
  box.append(figure(r.denoised, "denoised " + r.name, r), effect(r));
  return box;
}

function gate(r) {
  const box = el("div", "gate " + (r.photo ? "yes" : "no"));
  const meter = el("div", "meter"), fill = el("span");
  fill.style.width = (100 * r.p_photo).toFixed(1) + "%";
  meter.append(fill);
  box.append(el("p", "prob", "P(photo) = " + r.p_photo.toFixed(3)), meter,
             el("p", "verdict", r.photo ? (r.denoise_first ? "Photo: sent on to captioning" : "Photo: sent on to pre-processing")
                     : "Not a photo: the workflow stops"));
  return box;
}

function fill(row, r, version) {
  row.className = "row";
  row.replaceChildren();
  const head = el("div", "row-head");
  head.append(el("span", "num", "#" + row.dataset.n), el("span", "name", r.name));
  if (!r.error) head.append(el("span", "order-tag", r.denoise_first ? "2 → 1 → 3" : "1 → 2 → 3"));
  if (r.size) head.append(el("span", "dim", r.size + " px · " + r.seconds + " s for both versions"));
  row.append(head);
  if (r.error) {
    row.classList.add("error");
    row.append(el("p", "error-text", r.error));
    return;
  }
  const chain = el("div", "chain");
  const inputStep = step("Input", "Unidentified image", figure(r.input, "input " + r.name, r));
  const d1 = step("1 · Binary classification", "Photo or not?", gate(r));
  const d2 = step("2 · Pre-processing", "Denoised",
                  r.denoised ? denoisedStep(r) : el("p", "skipped", "Not run: only photographs are cleaned"));
  const caption = r.captions[version];
  const d3 = step("3 · Captioning", "Caption",
                  caption ? el("p", "caption", caption) : el("p", "caption none", "No caption"));
  chain.append(...(r.denoise_first ? [inputStep, d2, d1, d3] : [inputStep, d1, d2, d3]));
  row.append(chain);
}

function placeholder(n, name) {
  const row = el("li", "row pending");
  row.dataset.n = n;
  const head = el("div", "row-head");
  head.append(el("span", "num", "#" + n), el("span", "name", name), el("span", "dim", "processing…"));
  row.append(head);
  return row;
}

function updateStatus() {
  const c = counts, skipped = +statusLine.dataset.skipped || 0;
  if (!c.total) {
    statusLine.textContent = skipped ? `No image yet · ${skipped} non-image files skipped.` : "No image yet.";
    clearButton.hidden = !skipped; return;
  }
  const parts = [`${c.done} of ${c.total} processed`, `${c.photos} captioned`, `${c.rejected} not photos`];
  if (c.errors) parts.push(`${c.errors} unreadable`);
  if (skipped) parts.push(`${skipped} non-image files skipped`);
  statusLine.textContent = parts.join(" · ");
  clearButton.hidden = busy;
}

const IMAGE_EXT = /\.(jpe?g|png|webp|bmp|tiff?|gif)$/i;
const isImage = f => f.type.startsWith("image/") || IMAGE_EXT.test(f.name);

// A dropped folder arrives as an unreadable pseudo-file: walk it (and its sub-folders) to get the real files.
function readEntry(entry) {
  if (entry.isFile) return new Promise(ok => entry.file(ok, () => ok(null)));
  if (!entry.isDirectory) return Promise.resolve([]);
  const reader = entry.createReader(), all = [];
  return new Promise(ok => {
    const batch = () => reader.readEntries(async list => {
      if (!list.length) return ok((await Promise.all(all.map(readEntry))).flat());
      all.push(...list); batch();
    }, () => ok([]));
    batch();
  });
}

async function filesFromDrop(dt) {
  const entries = [...(dt.items || [])].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  const files = (await Promise.all(entries.map(readEntry))).flat().filter(Boolean);
  return files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
}

function add(all) {
  all = [...all];
  const files = all.filter(isImage), skipped = all.length - files.length;
  if (skipped) statusLine.dataset.skipped = (+statusLine.dataset.skipped || 0) + skipped;
  if (!files.length) { updateStatus(); return; }
  for (const f of files) {
    counts.total += 1;
    const rows = VERSIONS.map(v => {
      const row = placeholder(counts.total, f.name);
      document.querySelector(`#v-${v} .rows`).append(row);
      return row;
    });
    queue.push({ file: f, rows, order });
  }
  updateStatus();
  next();
}

async function next() {
  if (busy || !queue.length) return;
  busy = true; updateStatus();
  const { file, rows, order: o } = queue.shift();
  let r;
  let body;
  try {
    body = await file.arrayBuffer();               // read locally first, so a file error is not blamed on the server
  } catch (e) {
    r = { name: file.name, error: "the browser could not read this file (a folder, or a file that moved)" };
  }
  if (body) {
    try {
      const res = await fetch("/process", { method: "POST", body, headers: { "X-Filename": encodeURIComponent(file.name), "X-Order": o } });
      r = res.ok ? await res.json() : { name: file.name, error: `server error ${res.status}` };
    } catch (e) {
      r = { name: file.name, error: "the server did not answer (is start.bat still running?)" };
    }
  }
  rows.forEach((row, i) => fill(row, r, VERSIONS[i]));
  counts.done += 1;
  if (r.error) counts.errors += 1; else if (r.photo) counts.photos += 1; else counts.rejected += 1;
  busy = false; updateStatus();
  next();
}

function setOrder(o) {
  order = o;
  orderSwitch.classList.toggle("alt", o === "213");
  orderSwitch.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.order === o)));
  orderNote.textContent = NOTES[o];
  const flow = document.getElementById("flow"), one = document.getElementById("flow-1"), two = document.getElementById("flow-2");
  if (o === "213") flow.insertBefore(two, one); else flow.insertBefore(one, two);
  try { localStorage.setItem("leyenda-order", o); } catch (e) {}
}

orderSwitch.querySelectorAll("button").forEach(b => b.addEventListener("click", () => setOrder(b.dataset.order)));
let saved = null;
try { saved = localStorage.getItem("leyenda-order"); } catch (e) {}
setOrder(NOTES[saved] ? saved : "123");

input.addEventListener("change", () => { add(input.files); input.value = ""; });
["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async e => add(await filesFromDrop(e.dataTransfer)));
clearButton.addEventListener("click", () => {
  document.querySelectorAll(".rows").forEach(r => r.replaceChildren());
  delete statusLine.dataset.skipped;
  counts = { total: 0, done: 0, photos: 0, rejected: 0, errors: 0 };
  updateStatus();
});
