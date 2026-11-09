"use strict";
// Renders benchmark records exported by ai_inference_lab. No numbers are hardcoded here.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v, d = 2) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));

function lineChart(title, ylabel, series) {
  // series: [{name, dash, points:[[x,y],...]}]; x = batch size on a log2 axis
  const W = 440, H = 280, m = {l: 54, r: 16, t: 34, b: 44};
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const ymax = Math.max(...series.flatMap((s) => s.points.map((p) => p[1]))) * 1.1 || 1;
  const X = (x) => m.l + (xs.length < 2 ? 0.5 : (Math.log2(x) - Math.log2(xs[0])) / (Math.log2(xs[xs.length - 1]) - Math.log2(xs[0]))) * (W - m.l - m.r);
  const Y = (y) => H - m.b - (y / ymax) * (H - m.t - m.b);
  let g = `<text x="${m.l}" y="18" style="font-weight:600;font-size:13px">${esc(title)}</text>`;
  for (let i = 0; i <= 4; i++) {
    const v = (ymax * i) / 4;
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e5e7eb"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(v < 10 ? 1 : 0)}</text>`;
  }
  xs.forEach((x) => (g += `<text x="${X(x)}" y="${H - m.b + 16}" text-anchor="middle">${x}</text>`));
  g += `<text x="${(W + m.l) / 2}" y="${H - 8}" text-anchor="middle">Batch size</text>`;
  g += `<text transform="translate(12 ${H / 2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text>`;
  series.forEach((s, i) => {
    const d = s.points.map((p, j) => `${j ? "L" : "M"}${X(p[0])},${Y(p[1])}`).join(" ");
    g += `<path d="${d}" fill="none" stroke="#111827" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ""}/>`;
    s.points.forEach((p) => (g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.5" fill="#111827"/>`));
    g += `<text x="${W - m.r}" y="${m.t + 4 + i * 16}" text-anchor="end">${esc(s.name)} ${s.dash ? "- -" : "—"}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${esc(title)}">${g}</svg>`;
}

function render(payload, label) {
  const recs = (payload.records || []).filter((r) => r.metrics && Object.keys(r.metrics).length && !(r.errors || []).length);
  $("src").textContent = `${label} (schema ${payload.schema_version})`;
  $("empty").hidden = recs.length > 0;
  if (!recs.length) return;
  const hw = recs[0].hardware;
  $("hw").innerHTML = [hw.device_name, `device: ${hw.device_type}`, `PyTorch ${hw.pytorch_version}`, `runtime: ${hw.runtime_environment}`, `CUDA: ${hw.cuda_available ? hw.cuda_version : "not available"}`, `precision: ${recs[0].precision}`, `model: ${recs[0].model.name}`, `warm-up ${recs[0].config.warmup_iters} / iters ${recs[0].config.measure_iters}`]
    .map((t) => `<span class="chip">${esc(t)}</span>`).join("");
  const key = (r) => `${r.hardware.device_type}/${r.precision}`;
  const groups = {};
  recs.forEach((r) => (groups[key(r)] = groups[key(r)] || []).push(r));
  const firstKey = Object.keys(groups)[0];
  const rows = groups[firstKey].sort((a, b) => a.batch_size - b.batch_size);
  $("c-lat").innerHTML = lineChart(`Batch size vs latency [${firstKey}]`, "Latency (ms)", [
    {name: "per batch", points: rows.map((r) => [r.batch_size, r.metrics.mean_ms])},
    {name: "per image", dash: true, points: rows.map((r) => [r.batch_size, r.metrics.per_image_mean_ms])}]);
  $("c-thr").innerHTML = lineChart(`Batch size vs throughput [${firstKey}]`, "Throughput (images/s)", [
    {name: "throughput", points: rows.map((r) => [r.batch_size, r.metrics.throughput_images_per_sec])}]);
  const head = ["device/precision", "batch", "mean ms", "median ms", "P90 ms", "P95 ms", "std ms", "per-image ms", "images/s", "GPU peak MB"];
  const body = recs.sort((a, b) => key(a).localeCompare(key(b)) || a.batch_size - b.batch_size).map((r) => {
    const m = r.metrics;
    return `<tr><td>${esc(key(r))}</td><td>${r.batch_size}</td><td>${fmt(m.mean_ms)}</td><td>${fmt(m.median_ms)}</td><td>${fmt(m.p90_ms)}</td><td>${fmt(m.p95_ms)}</td><td>${fmt(m.std_ms)}</td><td>${fmt(m.per_image_mean_ms)}</td><td>${fmt(m.throughput_images_per_sec, 1)}</td><td>${fmt(m.gpu_peak_allocated_mb, 0)}</td></tr>`;
  }).join("");
  $("table").innerHTML = `<div class="table-wrap"><table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;
}

$("file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try { render(JSON.parse(await f.text()), `你上傳的檔案:${f.name}`); }
  catch (err) { $("empty").hidden = false; $("empty").textContent = "無法解析 JSON:" + err.message; }
});

fetch("data/reference_cpu.json")
  .then((r) => r.json())
  .then((p) => render(p, "data/reference_cpu.json(CPU 預錄參考結果,非 live)"))
  .catch(() => { $("src").textContent = "(未載入)"; $("empty").hidden = false; });
