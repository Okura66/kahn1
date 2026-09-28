/* Kahn1 site: consent banner, the sidebar and its "on this page"
   list, the menu on small screens, and the home page's logo and reveals. The
   page's text lives in its HTML (one file per language); this script adds no copy. */
(() => {
  const el = (id) => document.getElementById(id);
  const root = document.documentElement;
  root.classList.add("js");
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
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
      a.onclick = (e) => { e.preventDefault(); setMenu(false); el("consent").hidden = false; el("consentYes").focus(); };
    }
    if (!["granted", "denied"].includes(store.get("consent", ""))) el("consent").hidden = false;
  }

  /* Menu (small screens): the sidebar, full screen. */
  function setMenu(open) { root.classList.toggle("menu-open", open); }
  for (const b of document.querySelectorAll("[data-menu]")) b.addEventListener("click", () => setMenu(b.dataset.menu === "open"));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setMenu(false); });

  /* "On this page": one entry per section of <main> that has an id and a title.
     data-nav overrides the label; a leading "01 · " is dropped. */
  const list = document.querySelector(".otp-list");
  const secs = [...document.querySelectorAll("main section[id]")].filter((s) => s.dataset.nav || s.querySelector("h2"));
  const links = [];
  let fill = null, dot = null, rail = null;
  if (list && secs.length > 1) {  // a single section needs no list
    rail = list.appendChild(document.createElement("span")); rail.className = "track";
    fill = list.appendChild(document.createElement("span")); fill.className = "fill";
    dot = list.appendChild(document.createElement("span")); dot.className = "dot";
    for (const s of secs) {
      const a = list.appendChild(document.createElement("a"));
      a.href = "#" + s.id;
      a.textContent = s.dataset.nav || s.querySelector("h2").textContent.replace(/^\s*\d+\s*·\s*/, "").trim();
      a.addEventListener("click", () => setMenu(false));
      links.push(a);
    }
  } else if (list) list.closest(".otp").hidden = true;
  function track() {
    if (!links.length) return;
    const th = window.innerHeight * 0.4;
    const tops = secs.map((s) => s.getBoundingClientRect().top);
    let act = -1;
    tops.forEach((tp, k) => { if (tp < th) act = k; });
    let pos = 0;
    if (act >= 0) {
      const a = tops[act], b = act + 1 < tops.length ? tops[act + 1] : a + secs[act].offsetHeight;
      pos = Math.min(links.length - 1, act + Math.max(0, Math.min(1, (th - a) / Math.max(1, b - a))));
    }
    links.forEach((l, k) => l.classList.toggle("on", k === act));
    // Link centres (rows wrap, so their heights differ); the dot slides between them.
    const c = links.map((l) => l.offsetTop + l.offsetHeight / 2);
    const k = Math.floor(pos), y = c[k] + (pos - k) * ((c[k + 1] ?? c[k]) - c[k]);
    rail.style.top = fill.style.top = c[0] + "px";
    rail.style.height = c[c.length - 1] - c[0] + "px";
    dot.style.top = y.toFixed(1) + "px";
    dot.style.opacity = act >= 0 ? "1" : "0";
    fill.style.height = (y - c[0]).toFixed(1) + "px";
  }
  window.addEventListener("scroll", track, { passive: true });
  window.addEventListener("resize", track);
  track();

  /* Reveals: [data-reveal] rises in when it enters the view, staggered among
     siblings; .primset starts its bars; [data-decode] kickers unscramble. */
  if ("IntersectionObserver" in window && !still) {
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (!en.isIntersecting) return;
      en.target.classList.add(en.target.classList.contains("primset") ? "on" : "in");
      io.unobserve(en.target);
    }), { threshold: 0.2 });
    const groups = new Map();
    for (const r of document.querySelectorAll("[data-reveal]")) {
      const n = groups.get(r.parentElement) || 0;
      groups.set(r.parentElement, n + 1);
      r.classList.add("rv");
      r.style.transitionDelay = n * 0.08 + "s";
      io.observe(r);
    }
    for (const p of document.querySelectorAll(".primset")) io.observe(p);

    const GL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let pending = [...document.querySelectorAll("[data-decode]")];
    for (const d of pending) d.dataset.orig = d.textContent;
    const decode = (d) => {
      const txt = d.dataset.orig, t0 = performance.now();
      const run = (now) => {
        const t = (now - t0) / 1000, tick = Math.floor(t * 22);
        let out = "", done = true;
        for (let i = 0; i < txt.length; i++) {
          const ch = txt[i];
          if (ch === " " || t >= 0.15 + i * 0.035) { out += ch; continue; }
          done = false;
          out += GL[Math.floor(hash(tick * 31 + i * 7) * GL.length)];
        }
        d.textContent = out;
        if (!done) requestAnimationFrame(run);
      };
      requestAnimationFrame(run);
    };
    const check = () => {
      const vh = window.innerHeight;
      pending = pending.filter((d) => {
        const r = d.getBoundingClientRect();
        if (r.height > 0 && r.top < vh * 0.85 && r.bottom > 0) { decode(d); return false; }
        return true;
      });
      if (!pending.length) window.removeEventListener("scroll", check);
    };
    window.addEventListener("scroll", check, { passive: true });
    requestAnimationFrame(() => requestAnimationFrame(check));
  } else {
    for (const p of document.querySelectorAll(".primset")) p.classList.add("on");
  }

  function hash(n) { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }

  /* Home logo. Five slot letters settle out of noisy distributions over their
     candidates (a softmax that sharpens until it locks), the 1 glitches now and
     then, and on the first scroll the letters fly to the top-left corner. */
  const logo = el("k1-logo");
  if (!logo) { root.classList.add("docked"); return; }

  const INK = "#1c1b19", MUTED = "#8f897f", RULE = "#e4dfd6", ACCENT = "#d9542b";
  const END = 6.4, W = 1128;
  const SLOTS = [
    { win: "K", c: ["K", "X", "R", "H"] },
    { win: "A", c: ["4", "A", "R", "N"] },
    { win: "H", c: ["N", "K", "H", "M"] },
    { win: "N", c: ["M", "H", "W", "N"] },
    { win: "1", c: ["I", "L", "7", "1"] },
  ];
  const E = {
    enter: (x) => 1 - Math.pow(1 - x, 3),
    settle: (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2),
    pop: (x) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); },
  };
  const tw = (a, b, s, e, ease, t) => a + (b - a) * ease(Math.max(0, Math.min(1, (t - s) / (e - s))));
  const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };

  const mk = (parent, cls) => { const d = parent.appendChild(document.createElement("div")); if (cls) d.className = cls; return d; };
  const slots = SLOTS.map((S) => {
    const outer = mk(logo), slot = mk(outer, "k1-slot"), glyph = mk(slot, "k1-glyph");
    const bands = [mk(glyph), mk(glyph), mk(glyph)], scan = mk(glyph, "scan");
    const dist = mk(slot, "k1-dist");
    const bars = S.c.map((c, j) => {
      const b = dist.appendChild(document.createElement("b"));
      const l = dist.appendChild(document.createElement("i"));
      b.style.left = l.style.left = 14 + j * 46 + "px";
      l.textContent = c;
      return [b, l];
    });
    outer.addEventListener("click", () => window.scrollTo({ top: 0, behavior: still ? "auto" : "smooth" }));
    return { outer, slot, glyph, bands, scan, dist, bars };
  });
  const tag = logo.parentElement.querySelector(".tag h1"), facts = logo.parentElement.querySelector(".facts");
  const cue = logo.parentElement.querySelector(".cue");
  if (cue) cue.addEventListener("click", (ev) => { ev.preventDefault(); window.scrollTo({ top: window.innerHeight, behavior: still ? "auto" : "smooth" }); });

  const st = { t: still ? END : 0, pv: 0, target: 0, gt: null, gSeed: 0 };
  let t0 = performance.now(), tween = null, glitchT0 = null, raf = 0;

  function drawSlot(i, pi) {
    const S = SLOTS[i], D = slots[i], t = st.t;
    const wi = S.c.indexOf(S.win), lock = 1.0 + i * 0.28, locked = t >= lock;
    const sharp = tw(0, 4.6, 0.2, lock, E.settle, t), noise = tw(2.2, 0.4, 0, lock, E.enter, t);
    const tick = Math.floor(t * 11);
    const p = softmax(S.c.map((_, j) => (j === wi ? sharp : 0) + (hash(tick * 7 + i * 13 + j) - 0.5) * 2 * noise));
    let letter = locked ? S.win : S.c[p.indexOf(Math.max(...p))];
    let g = null;
    const lt = t >= 5.4 && t < 6.3 ? t - 5.4 : st.gt != null ? st.gt : -1;
    if (i === 4 && lt >= 0 && lt < 0.9) {
      const e = Math.sin((Math.PI * lt) / 0.9), tk = Math.floor((lt + st.gSeed) * 9);
      if (hash(tk + 1) > 0.45) letter = S.c[Math.floor(hash(tk) * 3)];
      g = { e, off: [0, 1, 2].map((k) => (hash(tk * 3 + k + 11) - 0.5) * 46 * e), split: (3 + hash(tk + 4) * 6) * e,
        dy: (hash(tk + 7) - 0.5) * 12 * e, sy: 1 - 0.06 * e * hash(tk + 2), flick: 1 - 0.3 * e * hash(tk + 8) };
    } else if (i === 4 && pi > 0.1 && pi < 0.9) {  // the 1 glitches in flight
      const e = Math.sin((Math.PI * (pi - 0.1)) / 0.8), tk = Math.floor(pi * 16) + 3;
      if (hash(tk + 1) > 0.5) letter = S.c[Math.floor(hash(tk) * 3)];
      g = { e, off: [0, 1, 2].map((k) => (hash(tk * 3 + k + 11) - 0.5) * 60 * e), split: (4 + hash(tk + 4) * 8) * e,
        dy: 0, sy: 1, flick: 1 - 0.25 * e * hash(tk + 8) };
    }
    const appear = tw(0, 1, 0.1 + i * 0.1, 0.6 + i * 0.1, E.enter, t);
    const popS = locked ? tw(1.08, 1, lock, lock + 0.45, E.pop, t) : 0.96;
    const distOp = tw(1, 0, 2.9, 4.1, E.settle, t);
    const color = locked ? (i === 4 ? ACCENT : INK) : MUTED;
    const cuts = g ? [[0, 38], [38, 64], [64, 100]] : [[0, 100]];
    D.bands.forEach((b, k) => {
      const c = cuts[k];
      b.hidden = !c;
      if (!c) return;
      if (b.textContent !== letter) b.textContent = letter;
      b.style.color = color;
      b.style.clipPath = g ? `inset(${c[0]}% -60px calc(${100 - c[1]}% - 1px) -60px)` : "none";
      b.style.transform = `translateX(${g ? g.off[k] : 0}px)`;
      b.style.textShadow = g ? `${g.split}px 0 rgba(28,27,25,0.55), ${-g.split}px 0 rgba(143,137,127,0.6)` : "none";
    });
    D.scan.style.opacity = g ? g.e * 0.9 : 0;
    D.slot.style.opacity = appear;
    D.glyph.style.opacity = g ? g.flick : 1;
    D.glyph.style.transform = `translateY(${(1 - appear) * 24 + (g ? g.dy : 0)}px) scale(${popS}) scaleY(${g ? g.sy : 1})`;
    D.dist.hidden = distOp <= 0.001;
    if (!D.dist.hidden) {
      D.dist.style.opacity = distOp;
      D.bars.forEach(([b, l], j) => {
        const bh = Math.max(2, p[j] * 130), isWin = S.c[j] === S.win;
        b.style.top = 450 - bh + "px";
        b.style.height = bh + "px";
        b.style.background = isWin && locked ? ACCENT : isWin ? "#bdb6ab" : RULE;
        l.style.color = isWin && locked ? INK : MUTED;
      });
    }
  }

  function draw() {
    const vw = root.clientWidth, vh = window.innerHeight, side = vw >= 900, t = st.t, p = st.pv;
    const sh = Math.min(1, (vw * 0.8) / W), sn = 0.1;
    const hx = (vw - W * sh) / 2, hy = vh * 0.42 - 150 * sh;
    SLOTS.forEach((_, i) => {
      // staggered travel: the 1 leads, the K follows last; each arcs up and stretches mid-flight
      const pi = Math.max(0, Math.min(1, (p - (4 - i) * 0.07) / 0.72));
      const ei = E.settle(pi), arc = Math.sin(Math.PI * pi);
      const s = sh + (sn - sh) * ei;
      const x0 = hx + i * 232 * sh, x1 = 32 + i * 232 * sn, x = x0 + (x1 - x0) * ei;
      const baseY = hy + ((side ? 28 : 15) - hy) * ei;
      const y = baseY - arc * Math.max(0, Math.min(60 * (1 + i * 0.25), (baseY - 8) * 0.6));
      const rot = -arc * (6 - i * 2);
      slots[i].outer.style.transform =
        `translate(${x}px, ${y}px) scale(${s}) rotate(${rot}deg) scale(${1 + 0.35 * arc}, ${1 - 0.18 * arc})`;
      drawSlot(i, pi);
    });
    logo.classList.toggle("docked", p > 0.5);
    root.classList.toggle("docked", p > 0.5);
    root.style.setProperty("--p", p.toFixed(3));
    root.style.setProperty("--pe", E.settle(p).toFixed(3));
    const fade = (k) => Math.max(0, 1 - p * k);
    if (tag) {
      tag.style.opacity = tw(0, 1, 4.0, 4.8, E.enter, t) * fade(2.2);
      tag.style.transform = `translateY(${(1 - tw(0, 1, 4.0, 4.8, E.enter, t)) * 16}px)`;
    }
    if (facts) facts.style.opacity = tw(0, 1, 4.4, 5.0, E.enter, t) * fade(2.2);
    if (cue) {
      const o = tw(0, 1, 5.0, 5.6, E.enter, t) * fade(4);
      cue.style.opacity = o;
      cue.style.pointerEvents = o > 0.2 ? "auto" : "none";
    }
  }

  function frame(now) {
    raf = 0;
    if (st.t < END) st.t = Math.min(END, (now - t0) / 1000);
    if (tween) {
      const k = tween.dur ? Math.min(1, (now - tween.t0) / tween.dur) : 1;
      st.pv = tween.from + (st.target - tween.from) * k;
      if (k >= 1) tween = null;
    }
    if (glitchT0 != null) {
      st.gt = (now - glitchT0) / 1000;
      if (st.gt >= 0.9) { st.gt = null; glitchT0 = null; scheduleGlitch(); }
    }
    draw();
    if (st.t < END || tween || glitchT0 != null) raf = requestAnimationFrame(frame);
    else if (!glitchTimer) scheduleGlitch();
  }
  const kick = () => { if (!raf) raf = requestAnimationFrame(frame); };

  let glitchTimer = 0;
  function scheduleGlitch() {
    if (still) return;
    clearTimeout(glitchTimer);
    glitchTimer = setTimeout(() => {
      if (document.hidden) { glitchTimer = 0; scheduleGlitch(); return; }
      st.gSeed = Math.floor(Math.random() * 1000);
      glitchT0 = performance.now();
      kick();
    }, 6000 + Math.random() * 5000);
  }

  function onScroll() {
    const target = window.scrollY > 24 ? 1 : 0;
    if (target === st.target) return;
    st.target = target;
    tween = { from: st.pv, t0: performance.now(), dur: still ? 0 : 950 * Math.abs(target - st.pv) };
    kick();
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", () => { draw(); });
  onScroll();
  // Opened mid-page (a #section link, a reload): no intro, logo already docked.
  const skipIntro = () => { if (window.scrollY > 24 && st.pv < 1) { st.pv = st.target = 1; tween = null; st.t = END; t0 = -1e9; draw(); } };
  skipIntro();
  window.addEventListener("load", skipIntro);
  draw();
  kick();
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
