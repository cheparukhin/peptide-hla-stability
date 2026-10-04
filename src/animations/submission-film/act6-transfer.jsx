// act6-transfer.jsx — Act VI, "what survived" (16 s), film-time 1:30–1:46.
//
// Not an ES module. This file is eval'd into one shared global scope after
// animations-v3.jsx and pmhc-scene-v2.jsx, so it borrows C, FONT, MOTION, tw,
// lerp, clamp and Easing by bare name and publishes window.ActTransfer. Every
// module-scope symbol here is prefixed AT6_ / AT6 so it can never collide with
// the intro scene or a sibling act — a duplicate top-level `const` in this
// loader is a SyntaxError, not a shadowing.
//
// THE BEAT. The acts before this one delivered a negative result: the expensive
// arm lost. This act turns to the thing nobody asked the cheap model for — a
// model trained only on peptide-HLA dissociation half-life, with no retraining
// of any kind, handed a *different assay* (peptides physically recovered from
// living cells by mass spectrometry) and separating them from human-proteome
// decoys at median AUROC 0.966. In the project site's words, that is "the
// evidence it learned real biology rather than dataset artefacts" — and the
// three controls, which take ten of the sixteen seconds, are what earn it.
//
// IT DOES NOT CLOSE THE FILM. The two ink lines and the credits moved to the
// act that follows (ActNext, film-time 1:46). This act ends held on the
// result, everything still on screen, for the next act to dissolve over.
//
// NOT AN OVERCLAIM, and the script must stay that way: elution is a selection
// effect with at least four filters besides stability, so this is evidence that
// the learned signal TRANSFERS, not a second measurement of stability accuracy.
// The caveat line ("Transfer to a different assay — not a second measurement
// of stability accuracy.") is load-bearing; do not cut it to buy screen time.
// The REPORT.md limitations disclosure that used to sit on this act's card is
// now ActNext's to carry.
//
// DATA CONTRACT — window.__FILM__.elution, read at render time, never hardcoded:
//   auroc_median, auroc_iqr_low, auroc_iqr_high, auprc_median, auprc_chance,
//   n_alleles, n_peptides, n_decoys, decoy_ratio,
//   controls.random_scores.auroc_median
//   controls.other_allele_ligand_decoys.auroc_median
//   controls.wrong_hla_pseudosequence.auroc_median
// Absent file, absent key or a non-finite value renders nothing rather than
// crashing; an absent control drops its own row only. Rounding happens at the
// point of display and nowhere else.
//
// THE CURVES ARE SCHEMATIC. No per-allele ROC points are in the data file, so
// each curve is the one-parameter family tpr = fpr^((1-A)/A), whose area is
// exactly A. The shape is therefore an illustration drawn to the measured
// AUROC, which the on-screen note says out loud. Every number, every bar length
// and the size of the collapse come from the data.

// ── timing, in act-local seconds ────────────────────────────────────────────
const AT6_T = {
  axes: 0.3, title: 0.55, head: 0.7, body: 1.15,
  curve: 1.05, curveEnd: 3.05,          // ROC draws; the AUROC counts up with it
  iqr: 2.95, event: 3.3, auprc: 3.6,
  tile: [4.05, 4.2, 4.35], caveat: 4.6,
  swap: 5.0,                            // beat 1 column out, ladder in
  ladder: 5.2,
  // THE ARGUMENT, one control at a time, 2.7 s apart. Each lands its bar, its
  // own ROC in the left panel, then its verdict — so the three read as a
  // sequence of claims rather than as a table.
  row: [5.7, 8.4, 11.1],                // random · other-allele · wrong groove
  verdict: [6.7, 9.4, 13.4],
  collapse: 12.5, collapseEnd: 13.15,   // the wrong row is built at the primary
  delta: 13.35,                         // value, held there, and only then falls
  caption: 13.8, claim: 14.9,
};


// ── geometry, stage px on 1920×1080 ─────────────────────────────────────────
const AT6_P = { x0: 200, x1: 760, y0: 860, y1: 300 };   // ROC plot box
const AT6_RX = 880;                                      // right column left edge
const AT6_LAD = { x0: 900, x1: 1780, axisY: 430, rowY: [500, 630, 760] };

const AT6_X = (f) => AT6_P.x0 + f * (AT6_P.x1 - AT6_P.x0);
const AT6_Y = (t) => AT6_P.y0 - t * (AT6_P.y0 - AT6_P.y1);
// AUROC → ladder x. 0.5 (chance) is the origin; 1.0 is the right edge.
const AT6_LX = (v) => AT6_LAD.x0 + Math.min(1, Math.max(0, (v - 0.5) / 0.5)) * (AT6_LAD.x1 - AT6_LAD.x0);

// ── data, resolved at render time ───────────────────────────────────────────
function AT6_data() {
  const F = typeof window === 'undefined' ? null : window.__FILM__;
  const E = F && F.elution;
  if (!E || typeof E !== 'object') return null;
  const n = (v) => (typeof v === 'number' && isFinite(v) ? v : null);
  const ctl = (k) => {
    const o = E.controls && E.controls[k];
    if (o == null) return null;
    return n(typeof o === 'number' ? o : o.auroc_median);
  };
  const primary = n(E.auroc_median);
  if (primary == null) return null;      // nothing to draw: render nothing
  return {
    primary,
    iqrLo: n(E.auroc_iqr_low), iqrHi: n(E.auroc_iqr_high),
    auprc: n(E.auprc_median), auprcChance: n(E.auprc_chance),
    nAlleles: n(E.n_alleles), nPeptides: n(E.n_peptides), nDecoys: n(E.n_decoys),
    ratio: n(E.decoy_ratio),
    random: ctl('random_scores'),
    other: ctl('other_allele_ligand_decoys'),
    wrong: ctl('wrong_hla_pseudosequence'),
  };
}

// display-only rounding
const AT6_f3 = (v) => (v == null ? '—' : v.toFixed(3));
const AT6_int = (v) => (v == null ? '—' : Math.round(v).toLocaleString('en-US'));

// tpr = fpr^((1-A)/A): the one-parameter ROC whose area is exactly A.
function AT6_roc(a) {
  const A = Math.min(0.9995, Math.max(0.5001, a));
  const k = (1 - A) / A;
  let d = '';
  for (let i = 0; i <= 180; i++) {
    const u = i / 180, f = u * u * u;          // dense sampling near fpr = 0
    d += (i === 0 ? 'M' : 'L') + AT6_X(f).toFixed(2) + ' ' + AT6_Y(Math.pow(f, k)).toFixed(2);
  }
  return d;
}

// ── small type helper: all type is HTML, all marks are SVG ──────────────────
function AT6Txt({ x, y, op, size, weight, color, align, ls, lh, width, wrap, children }) {
  if (op != null && op <= 0.001) return null;
  const tx = align === 'right' ? '-100%' : align === 'center' ? '-50%' : '0';
  return (
    <div style={{
      position: 'absolute', left: x, top: y, transform: `translate(${tx},0)`,
      opacity: op == null ? 1 : op,
      font: `${weight || 500} ${size || 24}px ${FONT}`,
      color: color || C.ink, letterSpacing: ls, lineHeight: lh,
      width: width, whiteSpace: wrap ? 'normal' : 'nowrap',
    }}>{children}</div>
  );
}

function AT6Wash({ op }) {
  if (op <= 0.001) return null;
  return <div style={{ position: 'absolute', inset: 0, opacity: op,
    background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)' }} />;
}

// ── beat 1 + 2 share the ROC panel on the left ──────────────────────────────
function AT6Plot({ t, d, L, drawP, ghosts }) {
  const ax = tw(t, AT6_T.axes, AT6_T.axes + 0.55, MOTION.draw);
  const gridOp = tw(t, AT6_T.axes + 0.2, AT6_T.axes + 0.7, Easing.linear);
  const chance = tw(t, AT6_T.axes + 0.35, AT6_T.axes + 0.9, Easing.linear);
  const w = AT6_P.x1 - AT6_P.x0, h = AT6_P.y0 - AT6_P.y1;
  return (
    <g>
      {[0.25, 0.5, 0.75].map((g) => (
        <g key={g} opacity={gridOp * 0.9}>
          <line x1={AT6_X(g)} y1={AT6_P.y0} x2={AT6_X(g)} y2={AT6_P.y1} stroke="#C9CCD4" strokeWidth="1" />
          <line x1={AT6_P.x0} y1={AT6_Y(g)} x2={AT6_P.x1} y2={AT6_Y(g)} stroke="#C9CCD4" strokeWidth="1" />
        </g>
      ))}
      <line x1={AT6_P.x0} y1={AT6_P.y0} x2={AT6_P.x0} y2={AT6_P.y0 - h * ax} stroke={C.ink} strokeWidth="2" />
      <line x1={AT6_P.x0} y1={AT6_P.y0} x2={AT6_P.x0 + w * ax} y2={AT6_P.y0} stroke={C.ink} strokeWidth="2" />
      <line x1={AT6_P.x0} y1={AT6_P.y0} x2={AT6_X(chance)} y2={AT6_Y(chance)}
        stroke={C.helix} strokeWidth="2" strokeDasharray="7 9" opacity={0.85} />
      {ghosts.map((g) => (
        <path key={g.key} d={AT6_roc(g.v)} fill="none" stroke={g.color} strokeWidth="2.5"
          strokeDasharray={g.dash} opacity={g.op} pathLength="1" />
      ))}
      <path d={AT6_roc(d.primary)} fill="none" stroke={C.ink} strokeWidth="4.5" strokeLinecap="round"
        pathLength="1" strokeDasharray="1" strokeDashoffset={1 - drawP} />
      {L && chance > 0.5 && (
        <circle cx={AT6_X(0.5)} cy={AT6_Y(0.5)} r="4" fill={C.helix} opacity={chance} />
      )}
    </g>
  );
}

// ── beat 2: the control ladder ──────────────────────────────────────────────
function AT6_rows(d) {
  const rows = [];
  if (d.random != null) rows.push({
    key: 'random', v: d.random, color: C.helix, dash: '6 7',
    label: 'random scores', note: 'the scoring setup itself, given random numbers',
    verdict: 'chance — the floor',
  });
  if (d.other != null) rows.push({
    key: 'other', v: d.other, color: '#1D3557', dash: null,
    label: "other HLA types' peptides as decoys", note: 'decoy peptides that are real ligands of other HLA types',
    verdict: 'not just "looks presentable"',
  });
  if (d.wrong != null) rows.push({
    key: 'wrong', v: d.wrong, color: C.crimson, dash: null, collapse: true,
    label: 'the WRONG HLA groove sequence', note: 'same peptides paired with the wrong HLA',
    // this row's verdict is the measured drop, drawn by the delta below
  });
  return rows;
}

function AT6Ladder({ t, d, L, op, rows }) {
  if (op <= 0.001) return null;
  const px = AT6_LX(d.primary);
  const axis = tw(t, AT6_T.ladder, AT6_T.ladder + 0.5, MOTION.draw);
  return (
    <g opacity={op}>
      <line x1={AT6_LAD.x0} y1={AT6_LAD.axisY} x2={AT6_LAD.x0 + (AT6_LAD.x1 - AT6_LAD.x0) * axis}
        y2={AT6_LAD.axisY} stroke={C.ink} strokeWidth="2" />
      {[0.5, 0.75, 1].map((v) => (
        <line key={v} x1={AT6_LX(v)} y1={AT6_LAD.axisY} x2={AT6_LX(v)} y2={AT6_LAD.axisY - 9}
          stroke={C.ink} strokeWidth="2" opacity={axis} />
      ))}
      <line x1={px} y1={AT6_LAD.axisY} x2={px} y2={AT6_LAD.rowY[2] + 56} stroke={C.ink} strokeWidth="1.5"
        strokeDasharray="5 7" opacity={0.42 * axis} />
      {rows.map((r, i) => {
        const y = AT6_LAD.rowY[i];
        const p = tw(t, AT6_T.row[i], AT6_T.row[i] + 0.45, MOTION.draw);
        if (p <= 0.001) return null;
        const v = r.shown;
        const x = lerp(AT6_LAD.x0, AT6_LX(v), p);
        return (
          <g key={r.key}>
            <rect x={AT6_LAD.x0} y={y} width={Math.max(0, x - AT6_LAD.x0)} height="12" fill={r.color} />
            <line x1={x} y1={y - 9} x2={x} y2={y + 21} stroke={C.ink} strokeWidth="2.5" />
            <rect x={AT6_RX} y={y - 48} width="14" height="14" fill={r.color} stroke={C.ink} strokeWidth="1.5" />
          </g>
        );
      })}
      {/* the collapse, drawn where the bar would have ended: a dashed stub from
          where it fell to the primary it was built at. */}
      {rows.map((r, i) => {
        if (!r.collapse || r.deltaOp <= 0.001) return null;
        const y = AT6_LAD.rowY[i] + 6;
        const a = AT6_LX(r.shown), b = px;
        return (
          <g key={'d' + r.key} opacity={r.deltaOp}>
            <line x1={a} y1={y} x2={b} y2={y} stroke={C.crimson} strokeWidth="2.5" strokeDasharray="7 7" />
            <line x1={b} y1={y - 12} x2={b} y2={y + 12} stroke={C.crimson} strokeWidth="2.5" />
          </g>
        );
      })}
    </g>
  );
}

// ── the act ─────────────────────────────────────────────────────────────────
function ActTransfer({ T, t0, L }) {
  const t = T - (t0 || 0);
  if (t < -0.3 || t > 16.6) return null;
  const d = AT6_data();
  if (!d) return null;                       // no data file: render nothing
  const labels = L !== false;

  const appear = tw(t, 0, 0.3, Easing.linear);
  const beats = appear;                      // nothing fades out: ActNext dissolves over this
  const col1 = 1 - tw(t, AT6_T.swap, AT6_T.swap + 0.3, Easing.linear);          // beat 1 column
  const col2 = tw(t, AT6_T.swap + 0.2, AT6_T.swap + 0.55, Easing.linear);       // beat 2 ladder

  // the curve draws and the headline number counts up on one ramp
  const drawP = tw(t, AT6_T.curve, AT6_T.curveEnd, Easing.easeOutCubic);
  const shownAuroc = lerp(0.5, d.primary, drawP);

  // the collapse: the wrong-groove row is built at the primary value, then
  // falls to its own. Both ends come from the data, so the drop is measured.
  const collapseP = tw(t, AT6_T.collapse, AT6_T.collapseEnd, Easing.easeInOutQuart);
  const rows = AT6_rows(d).map((r) => {
    if (!r.collapse) return Object.assign({}, r, { shown: r.v, deltaOp: 0 });
    return Object.assign({}, r, {
      shown: lerp(d.primary, r.v, collapseP),
      deltaOp: tw(t, AT6_T.delta, AT6_T.delta + 0.3, Easing.linear),
    });
  });

  // each control also draws its ROC into the left panel as it lands
  const ghosts = rows.map((r, i) => ({
    // the random control's ROC *is* the chance diagonal, so it would draw a
    // second line on top of one already there; the labelled diagonal carries it.
    key: r.key, v: r.shown, color: r.color, dash: r.dash, ghost: r.key !== 'random',
    op: tw(t, AT6_T.row[i], AT6_T.row[i] + 0.5, Easing.linear) * col2 * 0.95,
  })).filter((g) => g.ghost && g.op > 0.001);

  const wrongRow = rows.filter((r) => r.collapse)[0] || null;
  const drop = wrongRow ? d.primary - wrongRow.v : null;
  const fold = d.auprc != null && d.auprcChance ? d.auprc / d.auprcChance : null;

  const tiles = [
    [AT6_int(d.nPeptides), 'eluted ligands'],
    [AT6_int(d.nDecoys), 'proteome decoys'],
    [AT6_int(d.nAlleles), 'HLA alleles'],
  ];

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', fontFamily: FONT }}>
      <AT6Wash op={appear} />

      {/* ── marks ── */}
      {beats > 0.001 && (
        <svg width="1920" height="1080" style={{ position: 'absolute', inset: 0 }} opacity={beats}>
          <AT6Plot t={t} d={d} L={labels} drawP={drawP} ghosts={ghosts} />
          <AT6Ladder t={t} d={d} L={labels} op={col2} rows={rows} />
          {/* AUPRC against chance — the harder statistic at a 10:1 decoy ratio */}
          {col1 > 0.001 && d.auprc != null && d.auprcChance != null && (() => {
            const p = tw(t, AT6_T.auprc, AT6_T.auprc + 0.55, MOTION.draw) * col1;
            if (p <= 0.001) return null;
            const bx = AT6_RX + 170, bw = 500;
            return (
              <g opacity={col1}>
                <rect x={bx} y="802" width={bw * d.auprc * p} height="11" fill={C.ink} />
                <rect x={bx} y="832" width={Math.max(2, bw * d.auprcChance * p)} height="11" fill={C.helix} />
                <line x1={bx} y1="794" x2={bx} y2="851" stroke={C.ink} strokeWidth="1.5" opacity={0.5 * p} />
              </g>
            );
          })()}
        </svg>
      )}

      {/* ── beat 1: the result ── */}
      {beats > 0.001 && col1 > 0.001 && (
        <div style={{ position: 'absolute', inset: 0, opacity: beats * col1 }}>
          <AT6Txt x={AT6_P.x0} y={248} op={tw(t, AT6_T.title, AT6_T.title + 0.4, Easing.linear)}
            size={20} weight={600} ls="0.13em" color="#5A5C6B">
            ELUTED LIGANDS vs PROTEOME DECOYS
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={296} op={tw(t, AT6_T.head - 0.1, AT6_T.head + 0.3, Easing.linear)}
            size={20} weight={600} ls="0.15em" color="#5A5C6B">
            A SECOND ASSAY · NO RETRAINING
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={352} op={tw(t, AT6_T.head, AT6_T.head + 0.5, MOTION.enter)} size={46} weight={600}>
            Nobody asked it for this.
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={424} op={tw(t, AT6_T.body, AT6_T.body + 0.5, Easing.linear)}
            size={29} weight={400} lh="1.5" width={900} wrap color="#3C3E52">
            Trained only to predict how long peptides stay bound — and never
            retrained — it ranked peptides recovered from living cells above matched decoys.
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={576} op={tw(t, AT6_T.curve, AT6_T.curve + 0.4, Easing.linear)}
            size={108} weight={600} lh="1">
            {AT6_f3(shownAuroc)}
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={702} op={tw(t, AT6_T.iqr, AT6_T.iqr + 0.4, Easing.linear)} size={24} color="#3C3E52">
            median AUROC{d.nAlleles != null ? ` · ${AT6_int(d.nAlleles)} alleles` : ''}
            {d.iqrLo != null && d.iqrHi != null ? ` · IQR ${AT6_f3(d.iqrLo)}–${AT6_f3(d.iqrHi)}` : ''}
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={736} op={tw(t, AT6_T.event, AT6_T.event + 0.4, Easing.linear)} size={24} color="#6A6C7A">
            a different assay · a different biological event
          </AT6Txt>
          {d.auprc != null && (
            <React.Fragment>
              <AT6Txt x={AT6_RX} y={796} op={tw(t, AT6_T.auprc, AT6_T.auprc + 0.4, Easing.linear)} size={21} color="#3C3E52">
                AUPRC {AT6_f3(d.auprc)}
              </AT6Txt>
              <AT6Txt x={AT6_RX} y={826} op={tw(t, AT6_T.auprc + 0.1, AT6_T.auprc + 0.5, Easing.linear)} size={21} color="#6A6C7A">
                chance {AT6_f3(d.auprcChance)}
              </AT6Txt>
              {fold != null && (
                <AT6Txt x={1800} y={808} align="right" op={tw(t, AT6_T.auprc + 0.25, AT6_T.auprc + 0.65, Easing.linear)}
                  size={26} weight={600}>
                  {fold.toFixed(1)}× chance{d.ratio ? ` at ${Math.round(d.ratio)}:1` : ''}
                </AT6Txt>
              )}
            </React.Fragment>
          )}
          {tiles.map((tile, i) => (
            <React.Fragment key={i}>
              <AT6Txt x={AT6_RX + i * 310} y={888} op={tw(t, AT6_T.tile[i], AT6_T.tile[i] + 0.35, MOTION.enter)}
                size={42} weight={600}>{tile[0]}</AT6Txt>
              <AT6Txt x={AT6_RX + i * 310} y={940} op={tw(t, AT6_T.tile[i] + 0.08, AT6_T.tile[i] + 0.4, Easing.linear)}
                size={20} color="#6A6C7A">{tile[1]}</AT6Txt>
            </React.Fragment>
          ))}
          <AT6Txt x={AT6_RX} y={992} op={tw(t, AT6_T.caveat, AT6_T.caveat + 0.4, Easing.linear)}
            size={20} weight={400} color="#6A6C7A">
            Transfer to a different assay — not a second measurement of stability accuracy.
          </AT6Txt>
        </div>
      )}

      {/* ── beat 2: the controls ── */}
      {beats > 0.001 && col2 > 0.001 && (
        <div style={{ position: 'absolute', inset: 0, opacity: beats * col2 }}>
          <AT6Txt x={AT6_RX} y={296} size={20} weight={600} ls="0.15em" color="#5A5C6B">
            THE CONTROLS
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={336} size={34} weight={600}>
            Three ways this could have been fake.
          </AT6Txt>
          {labels && [0.5, 0.75, 1].map((v) => (
            <AT6Txt key={v} x={AT6_LX(v)} y={386} align="center" size={19} color="#6A6C7A">
              {v.toFixed(2)}
            </AT6Txt>
          ))}
          {rows.map((r, i) => {
            const p = tw(t, AT6_T.row[i], AT6_T.row[i] + 0.4, Easing.linear);
            if (p <= 0.001) return null;
            const y = AT6_LAD.rowY[i];
            return (
              <React.Fragment key={r.key}>
                <AT6Txt x={AT6_RX + 26} y={y - 52} op={p} size={26} weight={600}>{r.label}</AT6Txt>
                <AT6Txt x={AT6_LAD.x1} y={y - 56} align="right" op={p} size={30} weight={600}
                  color={r.collapse ? C.crimson : C.ink}>
                  {AT6_f3(r.shown)}
                </AT6Txt>
                {labels && (
                  <AT6Txt x={AT6_RX + 26} y={y + 30} op={p} size={19} color="#6A6C7A">{r.note}</AT6Txt>
                )}
                {r.verdict && (
                  <AT6Txt x={AT6_LAD.x1} y={y + 26} align="right" size={22} weight={600}
                    op={tw(t, AT6_T.verdict[i], AT6_T.verdict[i] + 0.4, Easing.linear)}>
                    {r.verdict}
                  </AT6Txt>
                )}
              </React.Fragment>
            );
          })}
          {wrongRow && drop != null && (
            <AT6Txt x={AT6_LX(d.primary) - 16} y={AT6_LAD.rowY[2] + 26} align="right"
              op={wrongRow.deltaOp} size={24} weight={600} color={C.crimson}>
              −{drop.toFixed(3)} AUROC
            </AT6Txt>
          )}
          <AT6Txt x={AT6_LAD.x1} y={AT6_LAD.rowY[2] + 62} align="right" size={19} color="#6A6C7A">
            primary {AT6_f3(d.primary)}
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={872} op={tw(t, AT6_T.caption, AT6_T.caption + 0.4, MOTION.enter)} size={34} weight={600}>
            It did not memorise peptides.
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={916} op={tw(t, AT6_T.caption + 0.18, AT6_T.caption + 0.58, MOTION.enter)} size={34} weight={600}>
            It learned the allele.
          </AT6Txt>
          <AT6Txt x={AT6_RX} y={972} op={tw(t, AT6_T.claim, AT6_T.claim + 0.5, Easing.linear)} size={25} color="#3C3E52">
            Evidence it learned real biology, not quirks of the training data.
          </AT6Txt>
          <AT6Txt x={AT6_P.x0} y={930} op={tw(t, AT6_T.ladder + 0.15, AT6_T.ladder + 0.55, Easing.linear)}
            size={20} color="#6A6C7A">
            {AT6_int(d.nPeptides)} ligands · {AT6_int(d.nDecoys)} decoys{d.ratio ? ` · ${Math.round(d.ratio)}:1` : ''}
          </AT6Txt>
        </div>
      )}

      {/* ── shared plot furniture, both beats ── */}
      {beats > 0.001 && labels && (
        <div style={{ position: 'absolute', inset: 0, opacity: beats }}>
          <AT6Txt x={AT6_P.x0} y={AT6_P.y0 + 12} align="center" size={19} color="#6A6C7A"
            op={tw(t, AT6_T.axes + 0.3, AT6_T.axes + 0.7, Easing.linear)}>0</AT6Txt>
          <AT6Txt x={AT6_P.x1} y={AT6_P.y0 + 12} align="center" size={19} color="#6A6C7A"
            op={tw(t, AT6_T.axes + 0.3, AT6_T.axes + 0.7, Easing.linear)}>1</AT6Txt>
          <AT6Txt x={AT6_P.x0 - 18} y={AT6_P.y1 - 12} align="right" size={19} color="#6A6C7A"
            op={tw(t, AT6_T.axes + 0.3, AT6_T.axes + 0.7, Easing.linear)}>1</AT6Txt>
          <AT6Txt x={AT6_X(0.80) + 10} y={AT6_Y(0.80) + 4} size={18} color="#8A8B90"
            op={tw(t, AT6_T.axes + 0.5, AT6_T.axes + 0.9, Easing.linear)}>chance</AT6Txt>
          <AT6Txt x={(AT6_P.x0 + AT6_P.x1) / 2} y={AT6_P.y0 + 42} align="center" size={21} color="#3C3E52"
            op={tw(t, AT6_T.axes + 0.4, AT6_T.axes + 0.8, Easing.linear)}>false positive rate</AT6Txt>
          <div style={{ position: 'absolute', left: AT6_P.x0 - 96, top: (AT6_P.y0 + AT6_P.y1) / 2,
            transform: 'translate(-50%,-50%) rotate(-90deg)', font: `500 21px ${FONT}`, color: '#3C3E52',
            whiteSpace: 'nowrap', opacity: tw(t, AT6_T.axes + 0.4, AT6_T.axes + 0.8, Easing.linear) }}>
            true positive rate
          </div>
          <AT6Txt x={AT6_P.x0} y={930} size={18} color="#8A8B90"
            op={tw(t, AT6_T.curveEnd, AT6_T.curveEnd + 0.5, Easing.linear) * col1}>
            curve shape matches the measured AUROC; statistics are per-allele medians
          </AT6Txt>
        </div>
      )}

    </div>
  );
}

window.ActTransfer = ActTransfer;
