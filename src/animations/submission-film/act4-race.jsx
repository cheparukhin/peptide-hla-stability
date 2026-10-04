// ══════════════════════════════════════════════════════════════════════════
// ACT 4 — THE RACE (24 s)
//
// Three feature arms start together. Two finish on a laptop. The third folds
// the whole cohort on GPUs and bills for it — and the number the act lands on
// is the failure count, which is zero.
//
// Artifact-runtime file: no import / no export, one shared scope. It assumes
// React, Easing, clamp and the intro scene's C / FONT / MOTION / tw / lerp are
// already evaluated, and that film-scene.jsx renders it as
// <ActRace T={T} t0={AT.race} L={labels} />. Every identifier is RACE_* /
// race* / Race* so nothing collides in that shared scope.
//
// The film shell draws the wash and the vignette around every act and cuts
// four frames of black immediately after it — so the last authored frame here
// is a held frame, ready for that cut. The shell stops labelling chapters at
// act III, so this act draws its own tag in the shell's style and position.
//
// Render from T only: no effects, no rAF, no Math.random, so a seeked frame is
// a deterministic render. That is the exporter's contract.
//
// EVERY NUMBER comes from window.__FILM__.compute at render time (folds,
// failures, nonzero_returncodes, shards, gpu_hours, gpu_type, wall_hours,
// usd_total, cost_status, profiles). Nothing about the run is written into
// this file; rounding happens only where a value is painted. If the data block
// is missing the act renders nothing rather than inventing a figure.
// ══════════════════════════════════════════════════════════════════════════

const RACE_DUR = 24;
const RACE_W = 1920, RACE_H = 1080;
const RACE_INK = '#2B2D42';
const RACE_DIM = '#6B6D7E';
const RACE_RULE = 'rgba(43,45,66,0.20)';
const RACE_CARD = 'rgba(249,250,252,0.95)';
const RACE_L = 160, RACE_R = RACE_W - 160;

// ── deterministic noise + display formatting (rounding lives here only) ──
const raceHash = (i) => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const raceGrp = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const raceInt = (v) => raceGrp(Math.max(0, Math.round(v)));
const raceUSD = (v) => {
  const c = Math.max(0, Math.round(v * 100));
  return '$' + raceGrp(Math.floor(c / 100)) + '.' + String(c % 100).padStart(2, '0');
};
const raceHM = (h) => {
  if (typeof h !== 'number' || !isFinite(h)) return null;
  const m = Math.max(0, Math.round(h * 60));
  return Math.floor(m / 60) + ' h ' + String(m % 60).padStart(2, '0') + ' min';
};
const raceDec = (v, d) => Math.max(0, v).toFixed(d);
const racePow = (k) => (t) => Math.pow(clamp(t, 0, 1), k);
const RACE_FOLD_E = racePow(0.72);   // quick ramp, long tail
const RACE_COST_E = racePow(0.95);   // near-linear accrual
const raceHHMM = (iso) => {
  const m = typeof iso === 'string' && iso.match(/T(\d{2}):(\d{2})/);
  return m ? m[1] + ':' + m[2] : null;
};

// ── the data contract ────────────────────────────────────────────────────
// Field names are window.__FILM__.compute's own; the extra aliases only make
// the act survive a rename of the generator, they are not a second source.
function raceData() {
  const film = (typeof window !== 'undefined' && window.__FILM__) || null;
  const c = film && film.compute;
  if (!c || typeof c !== 'object') return null;

  const num = (o, keys) => {
    if (!o) return null;
    for (let i = 0; i < keys.length; i++) {
      let v = o[keys[i]];
      if (v === undefined || v === null || v === '') continue;
      if (typeof v === 'string') { const s = v.replace(/[^0-9.\-]/g, ''); v = s === '' ? NaN : parseFloat(s); }
      if (typeof v === 'number' && isFinite(v)) return v;
    }
    return null;
  };
  const str = (o, keys) => {
    if (!o) return null;
    for (let i = 0; i < keys.length; i++) {
      const v = o[keys[i]];
      if (typeof v === 'string' && v.trim() !== '') return v.trim();
    }
    return null;
  };

  const folds = num(c, ['folds', 'folds_total', 'fold_count', 'n_folds']);
  const usd = num(c, ['usd_total', 'usd', 'cost_usd', 'spend_usd']);
  const gpu = num(c, ['gpu_hours', 'gpuHours', 'gpu_h']);
  const fail = num(c, ['failures', 'failed_total', 'execution_failures', 'failed']);
  const rc = num(c, ['nonzero_returncodes', 'nonzero_return_codes']);
  const shards = num(c, ['shards', 'shards_total']);
  const hours = num(c, ['wall_hours', 'wallHours']);
  if (folds === null && usd === null) return null;   // nothing to say — say nothing

  // Two real sub-streams: the cohort was split across two Modal profiles that
  // folded disjoint halves in parallel. Used only if both are fully populated.
  let profiles = null;
  const P = c.profiles;
  if (P && typeof P === 'object') {
    const keys = Object.keys(P);
    const rows = keys.map((k, i) => ({
      name: 'WORKER ' + String.fromCharCode(65 + i),   // never the real account name
      folds: num(P[k], ['folds']),
      usd: num(P[k], ['usd']),
      gpu: num(P[k], ['gpu_hours']),
      shards: num(P[k], ['shards']),
      sec: num(P[k], ['median_steady_fold_s']),
    })).filter((r) => r.folds !== null && r.folds > 0);
    if (rows.length >= 2) profiles = rows;
  }

  return {
    folds: folds, usd: usd, gpu: gpu, fail: fail, rc: rc, shards: shards,
    hours: hours, wall: raceHM(hours),
    gpuType: str(c, ['gpu_type']),
    derived: (str(c, ['cost_status']) || '').indexOf('derived') === 0,
    launched: raceHHMM(c.launched), completed: raceHHMM(c.completed),
    profiles: profiles,
  };
}

// Per-stream fill windows. Each stream runs its own share of the storm window,
// scaled by the GPU-hours it actually burned, so the slower worker lands last.
function raceStreams(D) {
  const rows = D.profiles || [{ name: null, folds: D.folds, usd: D.usd, gpu: D.gpu, shards: D.shards }];
  const w = rows.map((r) => (r.gpu !== null && r.gpu > 0 ? r.gpu : (r.folds || 1)));
  const wmax = Math.max.apply(null, w);
  return rows.map((r, i) => Object.assign({}, r, { end: RACE_F0 + (RACE_F1 - RACE_F0) * (w[i] / wmax) }));
}

// ── procedural complex glyphs ────────────────────────────────────────────
// Abstract ribbon forms on a unit box, built once. These stand for "a
// predicted complex"; nothing here depicts a real structure and no coordinate
// file is loaded.
const RACE_SHAPES = (function () {
  const out = [];
  for (let v = 0; v < 8; v++) {
    const ph = raceHash(v * 3 + 1) * 6.2832;
    const amp = 0.34 + 0.40 * raceHash(v * 7 + 2);
    const k = 1.3 + 1.9 * raceHash(v * 11 + 3);
    const tilt = (raceHash(v * 13 + 5) - 0.5) * 0.5;
    const pts = [];
    for (let s = 0; s <= 10; s++) {
      const u = s / 10;
      pts.push([(u - 0.5) * 1.9,
        amp * Math.sin(ph + u * Math.PI * k) * (0.45 + 0.55 * Math.sin(u * Math.PI)) + tilt * (u - 0.5)]);
    }
    let d = 'M ' + pts[0][0].toFixed(3) + ' ' + pts[0][1].toFixed(3);
    for (let s = 1; s < pts.length; s++) {
      const a = pts[s - 1], b = pts[s];
      d += ' Q ' + ((a[0] + b[0]) / 2).toFixed(3) + ' ' + a[1].toFixed(3)
        + ' ' + b[0].toFixed(3) + ' ' + b[1].toFixed(3);
    }
    out.push({ d: d, blob: pts[3] });
  }
  return out;
})();

const RACE_COLS = 20, RACE_ROWS = 11;
const RACE_GY0 = 272, RACE_GY1 = 872;
const RACE_CELLS = (function () {
  const out = [];
  const x0 = 64, x1 = RACE_W - 64;
  for (let r = 0; r < RACE_ROWS; r++) for (let c = 0; c < RACE_COLS; c++) {
    const i = r * RACE_COLS + c;
    out.push({
      i: i, col: c,
      cx: x0 + (x1 - x0) * (c + 0.5) / RACE_COLS + (raceHash(i * 2.3 + 5) - 0.5) * 18,
      cy: RACE_GY0 + (RACE_GY1 - RACE_GY0) * (r + 0.5) / RACE_ROWS + (raceHash(i * 3.7 + 9) - 0.5) * 16,
      v: Math.floor(raceHash(i * 5.1 + 2) * RACE_SHAPES.length) % RACE_SHAPES.length,
      a0: raceHash(i * 1.7 + 3) * 360,
      sp: (raceHash(i * 2.9 + 7) < 0.5 ? -1 : 1) * (9 + 26 * raceHash(i * 4.3 + 11)),
      ph: raceHash(i * 6.1 + 13) * 6.2832,
      fr: 1.5 + 2.6 * raceHash(i * 7.3 + 17),
      rr: 17 + 11 * raceHash(i * 8.9 + 19),
      key: raceHash(i * 9.7 + 23),
    });
  }
  return out;
})();

function RaceGlyph({ g, p, spinT, live, dim }) {
  const e = MOTION.pop(p);
  const r = g.rr * (0.35 + 0.65 * e);
  const sh = 0.5 + 0.5 * Math.sin(g.ph + spinT * g.fr);
  const op = clamp((0.30 + 0.44 * sh * live + 0.20 * (1 - live)) * p * dim, 0, 1);
  if (op <= 0.004 || r <= 0.2) return null;
  const S = RACE_SHAPES[g.v];
  return (
    <g opacity={op}
      transform={`translate(${g.cx.toFixed(2)} ${g.cy.toFixed(2)}) rotate(${(g.a0 + g.sp * spinT).toFixed(2)}) scale(${r.toFixed(3)})`}>
      <path d={S.d} fill="none" stroke={RACE_INK} strokeWidth={2.1 / r} strokeLinecap="round" />
      <path d={S.d} fill="none" stroke={RACE_INK} strokeWidth={1.2 / r} strokeLinecap="round"
        opacity="0.45" transform="translate(0 0.27)" />
      <circle cx={S.blob[0]} cy={S.blob[1]} r="0.18" fill={C.crimson} opacity="0.7" />
    </g>
  );
}

// ── timing ───────────────────────────────────────────────────────────────
const RACE_IN = [0.15, 0.50, 0.85];        // lanes drop in
const RACE_TICK = [5.45, 6.25];            // A and B land their ticks
const RACE_MORPH = [7.90, 8.70];           // lanes → the storm's furniture
const RACE_F0 = 8.35, RACE_F1 = 19.20;     // fold counter window
const RACE_M0 = 8.70, RACE_M1 = 20.10;     // money window — deliberately longer
const RACE_LAND = 20.25;                   // the failure card

// ── lanes ────────────────────────────────────────────────────────────────
const RACE_PEP = 'SLLMWITQV';
const RACE_LANES = [
  { k: 'A', name: 'SEQUENCE', sub: 'peptide + 34 HLA binding-site residues' },
  { k: 'B', name: 'ESM-2 35M', sub: 'frozen embeddings from the same residues' },
  { k: 'C', name: 'BOLTZ-2', sub: 'one predicted 3-D structure per pair' },
];
const RACE_CY = [350, 540, 730];                         // lane centre lines
const RACE_HOME = [[RACE_L, 304], [RACE_L, 494], [RACE_L, 684]];
const RACE_PARK = [[712, 142, 0.60], [952, 142, 0.60], [RACE_L, 126, 0.84]];
const RACE_ART = 700;
const RACE_TRK = 1250, RACE_TRKW = 360;

function raceLaneState(tl, i) {
  const drop = tw(tl, RACE_IN[i], RACE_IN[i] + 0.9, MOTION.enter);
  const m = tw(tl, RACE_MORPH[0] + i * 0.06, RACE_MORPH[1] + i * 0.06, MOTION.draw);
  const home = RACE_HOME[i], park = RACE_PARK[i];
  return {
    x: lerp(home[0], park[0], m),
    y: lerp(home[1] + (1 - drop) * 54, park[1], m),
    s: lerp(1, park[2], m),
    op: drop,
    bg: m,                        // the card backing fades in only once parked
    subOp: i === 2 ? 1 : 1 - m,   // A and B keep just their name as chips
    rowOp: drop * (1 - m),        // art + track exist in lane mode only
  };
}

// ── lane "process" art ───────────────────────────────────────────────────
function RaceLetters({ x, y, build, drift }) {
  const w = 34, gap = 6;
  return (
    <g>
      {RACE_PEP.split('').map((ch, j) => {
        const p = clamp(build * 10 - j * 0.7, 0, 1);
        if (p <= 0.004) return null;
        const e = MOTION.pop(p);
        const cx = x + j * (w + gap), cy = y - drift * (8 + 14 * raceHash(j + 31));
        return (
          <g key={j} opacity={clamp(p, 0, 1) * (1 - drift)}
            transform={`translate(${(cx + w / 2).toFixed(2)} ${cy.toFixed(2)}) scale(${(0.5 + 0.5 * e).toFixed(3)}) translate(${-w / 2} 0)`}>
            <rect x="0" y="-17" width={w} height="34" fill="none" stroke={RACE_INK} strokeWidth="1.6" />
            <text x={w / 2} y="7" textAnchor="middle" fill={RACE_INK} style={{ font: `600 20px ${FONT}` }}>{ch}</text>
          </g>
        );
      })}
    </g>
  );
}

// (a) SEQUENCE — the peptide letters plus the 34 contact positions.
function RaceArtSequence({ tl, op }) {
  if (op <= 0.004) return null;
  const b = tw(tl, RACE_IN[0] + 0.7, RACE_IN[0] + 1.9, Easing.linear);
  const t = tw(tl, RACE_IN[0] + 1.5, RACE_IN[0] + 2.6, Easing.linear);
  const y = RACE_CY[0];
  return (
    <g opacity={op}>
      <RaceLetters x={RACE_ART} y={y - 22} build={b} drift={0} />
      {Array.from({ length: 34 }).map((u, j) => {
        const p = clamp(t * 40 - j, 0, 1);
        if (p <= 0.004) return null;
        return <rect key={j} x={RACE_ART + j * 10.6} y={y + 16} width="4" height={16 * p} fill={RACE_INK} opacity="0.68" />;
      })}
      <text x={RACE_ART + 376} y={y + 34} fill={RACE_DIM} opacity={t} style={{ font: `400 17px ${FONT}` }}>
        ×30 networks
      </text>
    </g>
  );
}

// (b) ESM-2 — the peptide dissolving into an embedding strip.
function RaceArtEsm({ tl, op }) {
  if (op <= 0.004) return null;
  const b = tw(tl, RACE_IN[1] + 0.7, RACE_IN[1] + 1.7, Easing.linear);
  const d = tw(tl, RACE_IN[1] + 1.8, RACE_IN[1] + 3.2, MOTION.draw);
  const y = RACE_CY[1];
  return (
    <g opacity={op}>
      <RaceLetters x={RACE_ART} y={y - 6} build={b} drift={d} />
      {Array.from({ length: 40 }).map((u, j) => {
        const p = clamp(d * 1.25 - j * 0.012, 0, 1);
        if (p <= 0.004) return null;
        const h = (10 + 26 * raceHash(j * 4.1 + 61)) * p;
        return <rect key={j} x={RACE_ART + j * 9.4} y={y + 2 - h / 2} width="6" height={h}
          fill={RACE_INK} opacity={(0.22 + 0.55 * raceHash(j * 2.7 + 13)) * p} />;
      })}
      <text x={RACE_ART + 386} y={y + 8} fill={RACE_DIM} opacity={d} style={{ font: `400 17px ${FONT}` }}>
        ×30 networks
      </text>
    </g>
  );
}

// (c) BOLTZ-2 — the peptide becoming a folded complex.
function RaceArtBoltz({ tl, op, total }) {
  if (op <= 0.004) return null;
  const b = tw(tl, RACE_IN[2] + 0.7, RACE_IN[2] + 1.7, Easing.linear);
  const d = tw(tl, RACE_IN[2] + 1.9, RACE_IN[2] + 3.6, MOTION.draw);
  const y = RACE_CY[2], S = RACE_SHAPES[2], r = 60;
  return (
    <g opacity={op}>
      <RaceLetters x={RACE_ART} y={y - 4} build={b} drift={d} />
      {d > 0.004 && (
        <g opacity={d}
          transform={`translate(${RACE_ART + 180} ${y}) rotate(${((tl - RACE_IN[2] - 2.6) * 14 * d).toFixed(2)}) scale(${(r * (0.55 + 0.45 * d)).toFixed(2)})`}>
          <path d={S.d} fill="none" stroke={RACE_INK} strokeWidth={2.4 / r} strokeLinecap="round"
            pathLength="1" strokeDasharray="1" strokeDashoffset={1 - d} />
          <path d={S.d} fill="none" stroke={RACE_INK} strokeWidth={1.4 / r} strokeLinecap="round" opacity="0.45"
            transform="translate(0 0.27)" pathLength="1" strokeDasharray="1" strokeDashoffset={1 - d} />
          <circle cx={S.blob[0]} cy={S.blob[1]} r="0.17" fill={C.crimson} opacity={0.8 * d} />
        </g>
      )}
      {total ? (
        <text x={RACE_ART + 376} y={y + 8} fill={RACE_DIM} opacity={d} style={{ font: `400 17px ${FONT}` }}>
          {raceInt(total) + ' folds'}
        </text>
      ) : null}
    </g>
  );
}

function RaceTrack({ y, op, prog, running, tl }) {
  if (op <= 0.004) return null;
  const w = RACE_TRKW * clamp(prog, 0, 1);
  return (
    <g opacity={op}>
      <rect x={RACE_TRK} y={y - 5} width={RACE_TRKW} height="10" fill="none" stroke={RACE_INK} strokeWidth="1.4" opacity="0.32" />
      <rect x={RACE_TRK} y={y - 5} width={w} height="10" fill={RACE_INK} />
      {running && (
        <rect x={RACE_TRK + w - 2} y={y - 9} width="3" height="18" fill={C.crimson}
          opacity={0.55 + 0.45 * Math.sin(tl * 5.2)} />
      )}
      <text x={RACE_TRK + RACE_TRKW + 24} y={y + 8} fill={RACE_INK}
        style={{ font: `500 22px ${FONT}`, fontVariantNumeric: 'tabular-nums' }}>
        {String(Math.floor(clamp(prog, 0, 1) * 100)) + '%'}
      </text>
    </g>
  );
}

// The lane label block is HTML so one transform carries the whole morph from
// lane row to parked chip.
function RaceLabel({ lane, st, done, tickP, blink }) {
  if (st.op <= 0.004) return null;
  return (
    <div style={{ position: 'absolute', left: 0, top: 0, opacity: st.op,
      transform: `translate(${st.x.toFixed(1)}px, ${st.y.toFixed(1)}px) scale(${st.s.toFixed(3)})`,
      transformOrigin: '0 0' }}>
      <div style={{ display: 'inline-block', padding: '10px 18px 12px 14px', borderRadius: 4,
        background: `rgba(249,250,252,${(0.95 * st.bg).toFixed(3)})`,
        border: `1.5px solid rgba(43,45,66,${(0.9 * st.bg).toFixed(3)})`, whiteSpace: 'nowrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ position: 'relative', width: 36, height: 36, flex: '0 0 36px', display: 'block',
            border: `1.6px solid ${done ? RACE_INK : C.crimson}` }}>
            {done ? (
              <svg width="36" height="36" style={{ position: 'absolute', left: 0, top: 0 }}>
                <path d="M 8 18.5 L 15.5 26 L 28.5 10" fill="none" stroke={RACE_INK} strokeWidth="3.4"
                  strokeLinecap="round" strokeLinejoin="round" pathLength="1" strokeDasharray="1"
                  strokeDashoffset={1 - clamp(tickP, 0, 1)} />
              </svg>
            ) : (
              <span style={{ position: 'absolute', left: 11, top: 11, width: 14, height: 14,
                background: C.crimson, opacity: 0.4 + 0.6 * Math.abs(Math.sin(blink * 3.1)) }} />
            )}
          </span>
          <span style={{ font: `600 20px ${FONT}`, color: RACE_DIM, letterSpacing: '0.16em' }}>{lane.k}</span>
          <span style={{ font: `600 40px ${FONT}`, color: RACE_INK, letterSpacing: '0.03em' }}>{lane.name}</span>
        </div>
        {st.subOp > 0.01 && (
          <div style={{ marginTop: 9, marginLeft: 52, font: `400 21px ${FONT}`, color: RACE_DIM,
            opacity: st.subOp, height: st.subOp < 0.99 ? 21 * st.subOp : undefined, overflow: 'hidden' }}>
            {lane.sub}
          </div>
        )}
      </div>
    </div>
  );
}

// ── readouts ─────────────────────────────────────────────────────────────
function RaceStat({ label, value, sub, size, strong, note }) {
  if (value === null || value === undefined) return null;
  return (
    <div style={{ padding: '14px 0 16px' }}>
      <div style={{ font: `500 17px ${FONT}`, color: RACE_DIM, letterSpacing: '0.20em' }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginTop: strong ? 0 : 4 }}>
        <span style={{ font: `${strong ? 600 : 500} ${size}px ${FONT}`, color: RACE_INK,
          letterSpacing: strong ? '-0.02em' : 0, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
        {sub ? <span style={{ font: `400 24px ${FONT}`, color: RACE_DIM, fontVariantNumeric: 'tabular-nums' }}>{sub}</span> : null}
      </div>
      {note ? <div style={{ marginTop: 2, font: `400 15px ${FONT}`, color: RACE_DIM, lineHeight: 1.35 }}>{note}</div> : null}
    </div>
  );
}

function RaceCaption({ tl, items }) {
  let cur = null;
  for (let i = 0; i < items.length; i++) if (tl >= items[i].at) cur = items[i];
  if (!cur) return null;
  const op = clamp((tl - cur.at) / 0.4, 0, 1);
  return (
    <div style={{ position: 'absolute', left: '7%', right: '7%', top: 898, opacity: op, textAlign: 'center',
      font: `400 32px ${FONT}`, color: RACE_INK, lineHeight: 1.4 }}>
      {cur.text}
      {cur.accent ? <span style={{ color: C.crimson }}> {cur.accent}</span> : null}
    </div>
  );
}

// ── the act ──────────────────────────────────────────────────────────────
function ActRace({ T, t0, L }) {
  const raw = T - t0;
  if (raw < 0) return null;
  const tl = Math.min(raw, RACE_DUR);     // hold the last frame for the hard cut
  const D = raceData();
  if (!D) return null;                    // no data block — render nothing

  const labels = L !== false;
  const streams = raceStreams(D);

  // Per-stream progress. Folds and GPU-hours accrue with the fold; the dollar
  // runs on a longer window, so the money is still climbing after the folds
  // have landed.
  const moneyTail = RACE_M1 - RACE_F1;
  const sp = streams.map((s) => ({
    s: s,
    p: tw(tl, RACE_F0, s.end, RACE_FOLD_E),
    m: tw(tl, RACE_M0, s.end + moneyTail, RACE_COST_E),
  }));
  const sum = (f) => sp.reduce((a, x) => a + f(x), 0);
  const foldsNow = sum((x) => (x.s.folds || 0) * x.p);
  const foldsTotal = sum((x) => x.s.folds || 0) || D.folds;
  const usdNow = D.usd === null ? null : (sp.every((x) => x.s.usd !== null && x.s.usd !== undefined)
    ? sum((x) => x.s.usd * x.m)
    : D.usd * tw(tl, RACE_M0, RACE_M1, RACE_COST_E));
  const gpuNow = D.gpu === null ? null : (sp.every((x) => x.s.gpu !== null && x.s.gpu !== undefined)
    ? sum((x) => x.s.gpu * x.p)
    : D.gpu * tw(tl, RACE_F0, RACE_F1, RACE_COST_E));
  const hoursNow = D.hours === null ? null : D.hours * tw(tl, RACE_F0, RACE_F1 + 0.5, Easing.linear);
  const cohort = foldsTotal ? clamp(foldsNow / foldsTotal, 0, 1) : 0;

  // storm
  const stormOn = tw(tl, RACE_MORPH[0] + 0.3, RACE_MORPH[1] + 0.5, Easing.linear);
  const spinT = tl <= 19.0 ? tl : 19.0 + 0.55 * (1 - Math.exp(-(tl - 19.0) / 0.55));
  const live = 1 - tw(tl, 19.4, 20.8, MOTION.draw);
  const dim = lerp(1, 0.28, tw(tl, 19.9, 21.0, MOTION.draw)) * stormOn;
  const perCol = RACE_COLS / sp.length;

  // lanes
  const lanes = [0, 1, 2].map((i) => raceLaneState(tl, i));
  const progA = tw(tl, 2.30, 5.25, MOTION.draw);
  const progB = tw(tl, 2.60, 6.05, MOTION.draw);
  const progC = clamp(0.03 * tw(tl, 2.90, RACE_F0, Easing.linear) + 0.97 * cohort, 0, 1);
  const tickA = tw(tl, RACE_TICK[0], RACE_TICK[0] + 0.45, MOTION.draw);
  const tickB = tw(tl, RACE_TICK[1], RACE_TICK[1] + 0.45, MOTION.draw);
  const tickC = tw(tl, RACE_LAND - 0.35, RACE_LAND + 0.15, MOTION.draw);

  const landP = tw(tl, RACE_LAND, RACE_LAND + 0.7, MOTION.enter);
  const panelP = tw(tl, RACE_MORPH[1] - 0.3, RACE_MORPH[1] + 0.5, MOTION.enter);
  const ruleP = tw(tl, 0.1, 0.8, Easing.linear);
  const tagP = tw(tl, 0.25, 0.9, Easing.linear);
  const mAvg = (lanes[0].bg + lanes[1].bg + lanes[2].bg) / 3;
  const ruleY = lerp(196, 232, mAvg);

  const caps = [
    { at: 0.80, text: 'Three approaches to features. One evaluation protocol.' },
    { at: 3.40, text: 'Two of them read sequence. One predicts a 3-D structure for every pair.' },
    { at: 5.70, text: 'The sequence models finish on a laptop, at no cost.' },
    { at: 8.90, text: foldsTotal
      ? 'Boltz-2 predicts a structure for every pair — ' + raceInt(foldsTotal) + ' complexes.'
      : 'Boltz-2 predicts a structure for every pair in the dataset.' },
    { at: 13.60, text: 'Two parallel workers, each folding its own half of the dataset.' },
    { at: 16.80, text: 'This is the single largest cost in the project.' },
    { at: RACE_LAND + 0.25, text: 'The pipeline did not fail.', accent: 'The features did not help.' },
  ];

  return (
    <div data-race-t={tl.toFixed(2)} style={{ position: 'absolute', inset: 0, overflow: 'hidden', fontFamily: FONT }}>
      {typeof FilmWash === 'function'
        ? <FilmWash />
        : <div style={{ position: 'absolute', inset: 0, background: '#E9EBEF' }} />}

      <svg width={RACE_W} height={RACE_H} style={{ position: 'absolute', inset: 0 }}>
        {/* instrument frame: hairlines only, no fills */}
        <line x1={RACE_L} y1={ruleY} x2={RACE_R} y2={ruleY} stroke={RACE_RULE} strokeWidth="1.5"
          pathLength="1" strokeDasharray="1" strokeDashoffset={1 - ruleP} />
        <line x1={RACE_L} y1="1026" x2={RACE_R} y2="1026" stroke={RACE_RULE} strokeWidth="1.5"
          pathLength="1" strokeDasharray="1" strokeDashoffset={1 - ruleP} opacity={1 - stormOn} />

        {/* 0–8 s: the three lanes */}
        <RaceArtSequence tl={tl} op={lanes[0].rowOp} />
        <RaceArtEsm tl={tl} op={lanes[1].rowOp} />
        <RaceArtBoltz tl={tl} op={lanes[2].rowOp} total={foldsTotal} />
        <RaceTrack y={RACE_CY[0]} op={lanes[0].rowOp} prog={progA} running={progA < 1} tl={tl} />
        <RaceTrack y={RACE_CY[1]} op={lanes[1].rowOp} prog={progB} running={progB < 1} tl={tl} />
        <RaceTrack y={RACE_CY[2]} op={lanes[2].rowOp} prog={progC} running tl={tl} />

        {/* 8–20 s: the fold storm, one band per worker */}
        {stormOn > 0.004 && RACE_CELLS.map((g) => {
          const si = Math.min(sp.length - 1, Math.floor(g.col / perCol));
          const p = clamp((sp[si].p * 1.08 - g.key) * 9, 0, 1);
          if (p <= 0.004) return null;
          return <RaceGlyph key={g.i} g={g} p={p} spinT={spinT} live={live} dim={dim} />;
        })}

        {/* the divider between the two disjoint halves */}
        {stormOn > 0.004 && sp.length > 1 && sp.slice(1).map((x, i) => {
          const xx = 64 + (RACE_W - 128) * ((i + 1) * perCol) / RACE_COLS;
          return <line key={i} x1={xx} y1={RACE_GY0 - 20} x2={xx} y2={RACE_GY1 + 16}
            stroke={RACE_RULE} strokeWidth="1.5" opacity={stormOn * 0.9} />;
        })}

        {/* lane C's track, grown into the cohort bar */}
        {stormOn > 0.004 && (
          <g opacity={stormOn}>
            <rect x={RACE_L} y="1012" width={RACE_R - RACE_L} height="12" fill="none" stroke={RACE_INK} strokeWidth="1.4" opacity="0.32" />
            <rect x={RACE_L} y="1012" width={(RACE_R - RACE_L) * clamp(progC, 0, 1)} height="12" fill={RACE_INK} />
            {progC < 0.999 && (
              <rect x={RACE_L + (RACE_R - RACE_L) * clamp(progC, 0, 1) - 2} y="1006" width="3" height="24"
                fill={C.crimson} opacity={0.5 + 0.5 * Math.sin(tl * 5.2)} />
            )}
          </g>
        )}
      </svg>

      {/* chapter tag — the shell hands this corner to the act from act III on */}
      {labels && tagP > 0.004 && (
        <div style={{ position: 'absolute', left: 80, top: 86, opacity: tagP, display: 'flex', gap: 16,
          alignItems: 'baseline', font: `500 22px ${FONT}`, color: RACE_INK, letterSpacing: '0.06em',
          textTransform: 'uppercase' }}>
          <span style={{ fontWeight: 600 }}>04</span>
          <span style={{ width: 36, height: 2, background: RACE_INK, alignSelf: 'center' }} />
          <span>the race, and the burn</span>
        </div>
      )}

      {/* lane labels, morphing into the storm's furniture */}
      <RaceLabel lane={RACE_LANES[0]} st={lanes[0]} done={tickA > 0} tickP={tickA} blink={tl} />
      <RaceLabel lane={RACE_LANES[1]} st={lanes[1]} done={tickB > 0} tickP={tickB} blink={tl} />
      <RaceLabel lane={RACE_LANES[2]} st={lanes[2]} done={tickC > 0} tickP={tickC} blink={tl} />

      {/* per-worker band labels */}
      {labels && stormOn > 0.004 && sp.length > 1 && sp.map((x, i) => (
        <div key={i} style={{ position: 'absolute', left: 64 + (RACE_W - 128) * (i * perCol) / RACE_COLS + 10,
          top: RACE_GY0 - 58, opacity: stormOn * 0.9 }}>
          <div style={{ font: `500 16px ${FONT}`, color: RACE_DIM, letterSpacing: '0.16em' }}>{x.s.name}</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginTop: 2 }}>
            <span style={{ font: `500 22px ${FONT}`, color: RACE_INK, fontVariantNumeric: 'tabular-nums' }}>
              {raceInt((x.s.folds || 0) * x.p)}
            </span>
            {x.s.folds !== null ? (
              <span style={{ font: `400 16px ${FONT}`, color: RACE_DIM, fontVariantNumeric: 'tabular-nums' }}>
                / {raceInt(x.s.folds)}
              </span>
            ) : null}
          </div>
        </div>
      ))}

      {/* the instrument panel: the folds, and then the money */}
      {panelP > 0.004 && (
        <div style={{ position: 'absolute', right: RACE_L, top: 74, width: 540, opacity: panelP,
          transform: `translateY(${((1 - panelP) * 14).toFixed(2)}px)`, borderRadius: 4, boxSizing: 'border-box',
          background: RACE_CARD, border: `1.5px solid ${RACE_INK}`, padding: '10px 26px 16px' }}>
          <RaceStat label="FOLDS COMPLETE" size={56}
            value={foldsTotal ? raceInt(foldsNow) : null}
            sub={foldsTotal ? '/ ' + raceInt(foldsTotal) : null} />
          <div style={{ height: 1.5, background: RACE_RULE }} />
          <RaceStat label={'SPEND · USD' + (D.derived ? ' (DERIVED)' : '')} size={104} strong
            value={usdNow === null ? null : raceUSD(usdNow)}
            note={D.derived ? 'derived from container hours and measured rate — not a direct bill' : null} />
          <div style={{ height: 1.5, background: RACE_RULE }} />
          <div style={{ display: 'flex', gap: 38, paddingTop: 14 }}>
            {gpuNow !== null && (
              <div>
                <div style={{ font: `500 15px ${FONT}`, color: RACE_DIM, letterSpacing: '0.18em' }}>GPU-HOURS</div>
                <div style={{ font: `500 30px ${FONT}`, color: RACE_INK, marginTop: 4, fontVariantNumeric: 'tabular-nums' }}>
                  {raceDec(gpuNow, 1)}{D.gpuType ? ' ' + D.gpuType : ''}
                </div>
              </div>
            )}
            {(hoursNow !== null || D.wall) && (
              <div>
                <div style={{ font: `500 15px ${FONT}`, color: RACE_DIM, letterSpacing: '0.18em' }}>WALL CLOCK</div>
                <div style={{ font: `500 30px ${FONT}`, color: RACE_INK, marginTop: 4, fontVariantNumeric: 'tabular-nums' }}>
                  {hoursNow !== null ? raceHM(hoursNow) : D.wall}
                </div>
                {landP > 0.004 && D.launched && D.completed && (
                  <div style={{ marginTop: 2, font: `400 15px ${FONT}`, color: RACE_DIM, opacity: landP,
                    fontVariantNumeric: 'tabular-nums' }}>{D.launched} → {D.completed}</div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* 20 s: everything lands, and the number that matters is zero */}
      {landP > 0.004 && D.fail !== null && (
        <div style={{ position: 'absolute', left: 200, top: 392, width: 960, opacity: landP, borderRadius: 4,
          transform: `translateY(${((1 - landP) * 18).toFixed(2)}px)`, boxSizing: 'border-box',
          background: RACE_CARD, border: `1.5px solid ${RACE_INK}`, padding: '30px 40px 36px',
          display: 'flex', alignItems: 'center', gap: 46 }}>
          <div style={{ flex: '0 0 auto' }}>
            <div style={{ font: `500 22px ${FONT}`, color: RACE_DIM, letterSpacing: '0.20em',
              whiteSpace: 'nowrap' }}>EXECUTION FAILURES</div>
            <div style={{ font: `600 168px ${FONT}`, color: RACE_INK, lineHeight: 1, marginTop: 4,
              letterSpacing: '-0.03em' }}>{raceInt(D.fail)}</div>
          </div>
          <div style={{ width: 1.5, alignSelf: 'stretch', background: RACE_RULE }} />
          <div style={{ font: `400 26px ${FONT}`, color: RACE_INK, lineHeight: 1.6,
            whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>
            {foldsTotal ? <div>{raceInt(foldsTotal)} folds attempted</div> : null}
            {foldsTotal ? <div>{raceInt(foldsTotal)} folds returned</div> : null}
            {D.shards !== null ? <div>{raceInt(D.shards)} / {raceInt(D.shards)} shards complete</div> : null}
            {D.rc !== null ? <div>{raceInt(D.rc)} non-zero exit codes</div> : null}
            {D.usd !== null ? <div style={{ color: RACE_DIM }}>{raceUSD(D.usd)} of GPU time</div> : null}
          </div>
        </div>
      )}

      {labels && <RaceCaption tl={tl} items={caps} />}

      {/* the cohort bar's own label row */}
      {stormOn > 0.004 && (
        <React.Fragment>
          <div style={{ position: 'absolute', left: RACE_L, top: 976, opacity: stormOn * 0.9,
            font: `500 16px ${FONT}`, color: RACE_DIM, letterSpacing: '0.20em' }}>COHORT FOLD PROGRESS</div>
          <div style={{ position: 'absolute', right: RACE_L, top: 970, opacity: stormOn,
            font: `500 22px ${FONT}`, color: RACE_INK, fontVariantNumeric: 'tabular-nums' }}>
            {Math.floor(clamp(progC, 0, 1) * 100)}%
          </div>
        </React.Fragment>
      )}
    </div>
  );
}

window.ActRace = ActRace;
