// Submission film — 2:00 integration shell.
//
// Mounts the existing pmhc-intro beats as Act I and the new acts around them on
// one continuous timeline. Not an ES module: this file, animations-v3.jsx and
// pmhc-scene-v2.jsx are all evaluated in one shared scope, runtime first, then
// the intro scene (whose helpers — C, FONT, MOTION, tw, lerp, Frame — this file
// reuses), then the act files, then this.
//
// Every act is a plain component taking {T, t0, L} and animating on local time
// T - t0, so acts stay independent and the running order lives only here.

const FILM_W = 1920, FILM_H = 1080;

// Authored running order. `dur` is the act's own length; `at` is derived.
// Act I's length is INTRO_LEN below, which must match the intro cue sum.
// Act I is trimmed from 38s to 14s for the results-first cut: the groove and
// the peptide locking in are kept, the TCR docking / activation / clonal
// expansion / body arc is dropped. A judge does not need the full immunology
// arc to read a ranking metric, and the deck gives biology one section.
const INTRO_CUES = {
  Groove: 1.8, Anatomy: 3.6, Candidates: 3.0, Lock: 1.4, Approach: 1.6,
  Interface: 1.2, Wobble: 0.6, Anchors: 0.3, Contact: 0.2, Activation: 0.1,
  Expansion: 0.1, Body: 0.1,
};

// Turn the duration map into absolute starts, the same way the runtime derives
// CUES from OM_SCENES, so the intro beats see the cue object they expect.
function introCueStarts(cues) {
  const A = {}; let acc = 0;
  Object.keys(cues).forEach((k) => { A[k] = acc; acc += cues[k]; });
  A.__total = acc;
  return A;
}
const INTRO_A = introCueStarts(INTRO_CUES);
const INTRO_LEN = INTRO_A.__total;

// Results-first cut, mapped onto the project site's section order
// (site/index.html): title -> the task -> our approach -> main result ->
// the caveat -> the one positive -> next steps. Totals 2:00 exactly.
// The data/firewall and fold-storm acts are deliberately not mounted: both are
// method, and the deck leads with results. Their files are kept on disk.
const ACTS = [
  { key: 'coldopen', dur: 6,  comp: () => window.ColdOpen },
  { key: 'intro',    dur: INTRO_LEN, comp: null },   // rendered inline below
  { key: 'question', dur: 12, comp: () => window.ActQuestion },
  { key: 'approach', dur: 12, comp: () => window.ActApproach },
  { key: 'verdict',  dur: 20, comp: () => window.ActVerdict },
  { key: 'transfer', dur: 16, comp: () => window.ActTransfer },
  { key: 'next',     dur: 14, comp: () => window.ActNext },
];

const AT = (() => {
  const m = {}; let acc = 0;
  ACTS.forEach((a) => { m[a.key] = acc; acc += a.dur; });
  m.__total = acc;
  return m;
})();

// The film's one hard cut: four frames of black immediately before the verdict.
const CUT_AT = AT.verdict, CUT_LEN = 4 / 30;

// Act I: the intro beats, shifted onto film time. They are defined in
// pmhc-scene-v2.jsx and take the cue object as `A`.
function ActIntro({ T, t0, L }) {
  const t = T - t0;
  if (t < -0.2 || t > INTRO_LEN + 0.4) return null;
  if (typeof BeatHLA !== 'function') return null;
  const A = INTRO_A;
  return (
    <React.Fragment>
      <BeatHLA T={t} A={A} L={L} />
      <BeatGroove T={t} A={A} L={L} />
      <BeatTCR T={t} A={A} L={L} />
      <BeatStability T={t} A={A} L={L} />
      <BeatContact T={t} A={A} L={L} />
      <BeatResponse T={t} A={A} L={L} />
      <BeatBody T={t} A={A} />
    </React.Fragment>
  );
}

// An act renders only inside its own window (plus a little overlap for
// cross-dissolves), so one act's stray frame can never sit under another's.
function ActSlot({ T, act, L }) {
  const t0 = AT[act.key];
  const Comp = act.comp ? act.comp() : null;
  if (typeof Comp !== 'function') return null;
  if (T < t0 - 0.3 || T > t0 + act.dur + 0.3) return null;
  return <Comp T={T} t0={t0} L={L} />;
}

// Only the biology act needs a tag from the shell. The headline card carries
// its own masthead and every act from the task beat onward draws its own
// header, which is what the guard below enforces.
const FILM_CHAPTERS = [
  ['coldopen', '', ''],
  ['intro',    '01', 'peptide-MHC interface'],
];

// Acts III-VI draw their own chapter headers at this same corner, so the shell
// only labels the acts that don't: the intro (whose own ChapterTag lives in
// Piece, which we don't mount) and the question act.
function FilmChapterTag({ T }) {
  if (T >= AT.question) return null;
  let cur = null;
  FILM_CHAPTERS.forEach((c) => { if (T >= AT[c[0]]) cur = c; });
  if (!cur || !cur[1]) return null;
  const since = T - AT[cur[0]];
  const op = clamp(since / 0.4, 0, 1) * (1 - clamp((T - AT.transfer - 7) / 0.6, 0, 1));
  if (op <= 0.001) return null;
  return (
    <div style={{ position: 'absolute', left: 80, top: 86, opacity: op, display: 'flex', gap: 16,
      alignItems: 'baseline', font: `500 22px ${FONT}`, color: C.ink, letterSpacing: '0.06em',
      textTransform: 'uppercase' }}>
      <span style={{ fontWeight: 600 }}>{cur[1]}</span>
      <span style={{ width: 36, height: 2, background: C.ink, alignSelf: 'center' }} />
      <span>{cur[2]}</span>
    </div>
  );
}

function Film({ labels, captions }) {
  const comp = useComposition();
  // window.__FILM_T__ pins the stage to one authored second, for frame review
  // and still export (index.html sets it from a ?t= query param). Playback is
  // unaffected when it is null.
  const T = (window.__FILM_T__ == null) ? comp.T : window.__FILM_T__;
  const L = labels !== false;
  const cut = T >= CUT_AT - CUT_LEN && T < CUT_AT ? 1 : 0;
  return (
    <div data-screen-label={`t=${Math.floor(T)}s`}
      style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: '#E9EBEF', fontFamily: FONT }}>
      <Wash />
      <ActIntro T={T} t0={AT.intro} L={L} />
      {ACTS.filter((a) => a.comp).map((a) => <ActSlot key={a.key} T={T} act={a} L={L} />)}
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 80% 75% at 50% 48%, rgba(43,45,66,0) 60%, rgba(43,45,66,0.14) 100%)' }} />
      <FilmChapterTag T={T} />
      {cut ? <div style={{ position: 'absolute', inset: 0, background: '#000' }} /> : null}
    </div>
  );
}

function SubmissionFilm() {
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);
  return (
    <React.Fragment>
      <CompositionStage width={FILM_W} height={FILM_H} scenes={window.OM_SCENES}
        playback={window.OM_PLAYBACK} bg="#E9EBEF">
        <Film labels={t.labels} captions={t.captions} />
      </CompositionStage>
      {window.__FILM_PANEL__ === false ? null : (
      <TweaksPanel>
        <TweakSection label="Overlays" />
        <TweakToggle label="Scientific labels" value={t.labels} onChange={(v) => setTweak('labels', v)} />
        <TweakToggle label="Captions" value={t.captions} onChange={(v) => setTweak('captions', v)} />
      </TweaksPanel>
      )}
    </React.Fragment>
  );
}

window.FILM_AT = AT;
window.SubmissionFilm = SubmissionFilm;
