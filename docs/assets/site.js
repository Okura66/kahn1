/* Kahn1 site: consent banner, view counter and the two edge tabs. The page's
   text lives in its HTML (one file per language); this script adds no copy. */
(() => {
  const el = (id) => document.getElementById(id);
  // Same keys as the playground, so one choice covers the whole site.
  const store = {
    get(k, d) { try { return localStorage.getItem("kahn1-demo:" + k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("kahn1-demo:" + k, v); } catch {} },
  };

  /* Consent. The choice itself is stored locally (strictly necessary, no consent needed). */
  function setConsent(granted) {
    store.set("consent", granted ? "granted" : "denied");
    el("consent").hidden = true;
    if (granted) { window.loadGTM && window.loadGTM(); return; }
    // Refusing after accepting: drop the Google Analytics cookies already set.
    for (const c of document.cookie.split(";")) {
      const name = c.split("=")[0].trim();
      if (/^_ga|^_gid|^_gat/.test(name)) {
        for (const domain of ["", location.hostname, "." + location.hostname.split(".").slice(-2).join(".")]) {
          document.cookie = name + "=; Max-Age=0; path=/" + (domain ? "; domain=" + domain : "");
        }
      }
    }
    if (window.__gtmLoaded) location.reload();  // stop the tags already running
  }
  if (el("consent")) {
    el("consentYes").onclick = () => setConsent(true);
    el("consentNo").onclick = () => setConsent(false);
    for (const a of document.querySelectorAll("[data-cookies]")) {
      a.onclick = (e) => { e.preventDefault(); el("consent").hidden = false; el("consentYes").focus(); };
    }
    if (!["granted", "denied"].includes(store.get("consent", ""))) el("consent").hidden = false;
  }

  /* Public view count, site-wide. GoatCounter caches this answer for up to 4 h. */
  const views = el("views");
  if (views) {
    fetch("https://kahn1.goatcounter.com/counter/TOTAL.json")
      .then((r) => (r.ok || r.status === 404 ? r.json() : Promise.reject()))
      .then((d) => {
        views.textContent = d.count + " " + (d.count === "1" ? views.dataset.one : views.dataset.many);
        views.hidden = false;
      })
      .catch(() => {});
  }

  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Edge tabs: a full card when the margin beside the content can hold it
     (wide screens), otherwise the vertical tab. */
  const edges = document.querySelectorAll(".edge");
  const wrap = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--wrap"), 10) || 1160;
  function placeEdges() {
    const roomy = (window.innerWidth - wrap) / 2 >= 300;
    for (const e of edges) e.classList.toggle("roomy", roomy);
  }
  if (edges.length) { window.addEventListener("resize", placeEdges); placeEdges(); }

  /* Snake card: a snake loops around a fixed path on a mini board. */
  const mini = el("mini");
  if (mini) {
    const n = 8, cells = [];
    for (let i = 0; i < n * n; i++) cells.push(mini.appendChild(document.createElement("i")));
    const path = [];  // a loop one cell in from the edge
    for (let x = 1; x < 7; x++) path.push([x, 1]);
    for (let y = 2; y < 7; y++) path.push([6, y]);
    for (let x = 5; x > 0; x--) path.push([x, 6]);
    for (let y = 5; y > 1; y--) path.push([1, y]);
    let t = 0;
    const draw = () => {
      for (const c of cells) c.className = "";
      for (let j = 0; j < 5; j++) {
        const [x, y] = path[(t - j + path.length) % path.length];
        cells[y * n + x].className = j === 0 ? "h" : "s";
      }
      const [fx, fy] = path[(t + 6) % path.length];
      cells[fy * n + fx].className = "f";
    };
    draw();
    if (!still) setInterval(() => { t = (t + 1) % path.length; draw(); }, 220);
  }

  /* Playground card: the distributions of the page's example answer, in turn.
     Frames come from the data-frames attribute (the recorded output). */
  const bars = el("minibars");
  if (bars && bars.dataset.frames) {
    const frames = JSON.parse(bars.dataset.frames);
    let f = 0;
    const draw = () => {
      const fr = frames[f], top = Math.max(...fr.rows.map((r) => r[1]));
      bars.innerHTML = "";
      const q = document.createElement("div");
      q.className = "q";
      q.textContent = fr.q;
      bars.appendChild(q);
      for (const [label, p] of fr.rows) {
        const r = document.createElement("div");
        r.className = "r" + (p === top ? " win" : "");
        r.innerHTML = "<span></span><span class=\"b\"><b></b></span><span></span>";
        r.children[0].textContent = label;
        r.children[2].textContent = (100 * p).toFixed(1) + "%";
        bars.appendChild(r);
        const b = r.querySelector("b");
        if (still) b.style.width = (100 * p) + "%";
        else requestAnimationFrame(() => requestAnimationFrame(() => { b.style.width = (100 * p) + "%"; }));
      }
    };
    draw();
    if (!still) setInterval(() => { f = (f + 1) % frames.length; draw(); }, 2600);
  }
})();

/* Logit reader: replays recorded Kahn1 answers step by step. The data (states,
   distributions, labels in the page's language) is in the page, in
   <script type="application/json" id="reader-data">; nothing is computed here. */
(() => {
  const root = document.getElementById("reader");
  const src = document.getElementById("reader-data");
  if (!root || !src) return;
  const { labels: L, demos } = JSON.parse(src.textContent);
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (sel) => root.querySelector(sel);
  const text = $(".rd-text"), cache = $(".rd-cache"), qs = $(".rd-qs");
  const passes = $("[data-passes]"), tokCount = $("[data-tokens]");
  const dots = [...root.querySelectorAll("[data-demo]")], pauseBtn = $("[data-pause]");
  let cur = 0, timers = [], paused = still;
  const later = (ms, fn) => timers.push(setTimeout(fn, still ? 0 : ms));
  const clear = () => { timers.forEach(clearTimeout); timers = []; };

  function build(d) {
    text.innerHTML = "";
    for (const w of d.state.split(" ")) {
      const s = document.createElement("span");
      s.className = "w";
      s.textContent = w + " ";
      text.appendChild(s);
    }
    cache.innerHTML = "";
    for (let i = 0; i < d.tokens; i++) cache.appendChild(document.createElement("i"));
    tokCount.textContent = d.tokens;
    qs.innerHTML = "";
    for (const q of d.questions) {
      const row = document.createElement("div");
      row.className = "rq";
      const top = Math.max(...q.cols.map((c) => c[1]));
      row.innerHTML =
        '<div class="rq-k"></div><div class="rq-h"><span class="cur"></span></div><div class="rq-o"></div><div class="rq-l"></div>';
      row.querySelector(".rq-k").innerHTML = "<span></span><span class=\"kind\"></span><small></small>";
      row.querySelector(".rq-k span").textContent = '"' + q.key + '"';
      row.querySelector(".rq-k .kind").textContent = q.kind.toUpperCase();
      row.querySelector(".rq-k small").textContent = q.q;
      const h = row.querySelector(".rq-h");
      q.cols.forEach(([tok, p]) => {
        const c = document.createElement("span");
        c.className = "c" + (p === top ? " win" : "");
        c.title = tok + " " + (100 * p).toFixed(1) + "%";
        c.innerHTML = "<span><b></b></span><i></i>";
        c.querySelector("i").textContent = tok;
        c.dataset.p = p;
        h.appendChild(c);
      });
      const o = row.querySelector(".rq-o");
      o.innerHTML = "<b></b><em></em>";
      o.querySelector("b").textContent = "→ " + q.out;
      o.querySelector("em").textContent = "p = " + (100 * top).toFixed(1) + "%";
      const leg = row.querySelector(".rq-l");
      q.cols.forEach(([tok, p], i) => {
        const part = document.createElement(p === top ? "b" : "span");
        part.textContent = (q.kind === "noul" ? "" : tok + " ") + q.names[i];
        leg.appendChild(part);
        if (i < q.cols.length - 1) leg.appendChild(document.createTextNode(" · "));
      });
      qs.appendChild(row);
    }
    passes.textContent = "0";
  }

  function play(i) {
    clear();
    cur = i;
    dots.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
    const d = demos[i];
    build(d);
    const words = [...text.children], cells = [...cache.children];
    const per = 26;  // ms per word of the state
    words.forEach((w, k) => later(k * per, () => {
      w.classList.add("on");
      const upto = Math.round(((k + 1) / words.length) * cells.length);
      for (let c = 0; c < upto; c++) cells[c].classList.add("on");
    }));
    let t = words.length * per + 350;
    [...qs.children].forEach((row, k) => {
      later(t, () => row.classList.add("on"));
      later(t + 380, () => {
        row.querySelector(".cur").remove();
        for (const c of row.querySelectorAll(".c")) {
          c.querySelector("b").style.height = Math.max(2, 100 * c.dataset.p) + "%";
        }
        passes.textContent = String(k + 1);
      });
      later(t + 900, () => row.querySelector(".rq-o").classList.add("on"));
      t += 1150;
    });
    if (!paused) later(t + 3600, () => play((cur + 1) % demos.length));
  }

  dots.forEach((b, j) => b.addEventListener("click", () => play(j)));
  if (pauseBtn) {
    const label = () => {
      pauseBtn.textContent = paused ? L.play : L.pause;
      pauseBtn.setAttribute("aria-pressed", String(paused));
    };
    pauseBtn.addEventListener("click", () => { paused = !paused; label(); if (!paused) play((cur + 1) % demos.length); else clear(); });
    if (still) pauseBtn.hidden = true;
    label();
  }
  // Start when the reader scrolls into view, so the first pass is seen.
  const io = "IntersectionObserver" in window ? new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); play(0); }
  }, { threshold: 0.3 }) : null;
  if (io) { build(demos[0]); io.observe(root); } else play(0);
})();

/* Copy buttons: data-copy="<id>" copies that element's text. */
(() => {
  for (const b of document.querySelectorAll("[data-copy]")) {
    const label = b.textContent;
    b.addEventListener("click", async () => {
      const text = document.getElementById(b.dataset.copy).textContent;
      try { await navigator.clipboard.writeText(text); b.textContent = b.dataset.done || "COPIED"; }
      catch { b.textContent = b.dataset.fail || "SELECT AND COPY"; }
      setTimeout(() => { b.textContent = label; }, 1800);
    });
  }
})();
