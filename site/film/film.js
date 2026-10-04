/* src/animations/pmhc-intro/animations-v3.jsx */
// @ds-adherence-ignore -- omelette starter scaffold (raw elements/hex/px by design)
// Copied omelette starter. Re-running copy_starter_component with this kind overwrites this file with the latest version (page content is unaffected).

/* BEGIN USAGE */
// animations-v3.jsx — continuous-composition animation engine.
//
// THE MODEL: the animation is ONE element tree rendered as a pure function
// of one authored-time axis. Nothing mounts or unmounts at section
// boundaries, so any element can move, morph, or persist across them by
// ordinary interpolation. The scene list (OM_SCENES) is the user-control
// view — names, order, playback durations — and the engine derives the cue
// table from it, so structure has exactly one source and cannot drift.
//
// API INDEX (every export is a window global):
//   <CompositionStage width height scenes={window.OM_SCENES}
//                     playback={window.OM_PLAYBACK} bg>
//     <Piece />   — ONE component, the whole animation
//   </CompositionStage>
//   useComposition() -> {T, CUES, time, duration, authoredTotal, playing}
//     T: authored seconds (warped per-section by user trims/speeds) —
//        key ALL choreography to T, never to wall-clock time
//     CUES: {SectionName: authoredStart} derived from OM_SCENES; an unknown
//        name returns NaN and raises a preview-only badge (never exports);
//        duplicate section names bind to the first occurrence
//   <Shot from={CUES.Build} to={CUES.Close}> — children visible between two
//     authored times (an authored hard cut in one line); children stay
//     mounted (media keeps its readiness) and are hidden outside the window
//   <Captions items={[{at, until?, text}, ...]} /> — ONE caption element,
//     at most one visible at a time, keyed to T; 'until' defaults to the
//     next item's 'at'; a last item with no 'until' stays to the end
//   WATERCOLOR (only when the Watercolor illustration skill is active —
//   otherwise ignore these entries). A painting is a function(p) written
//   against the paint kit, on a width x height sheet; it needs
//   watercolor_kit.js loaded by a <script> tag before this engine, must
//   be a stable function defined once (module scope, never an inline
//   arrow), and every component below renders <img> elements, so it all
//   exports by construction.
//   <WatercolorPainting painting={fn} from={CUES.X} to={CUES.Y} width height
//     seed scale quality style /> — the painting assembled from its own
//     STROKES: each wash / ink line / splatter is a separate layer
//     stacked over the paper, appearing in painting order between two
//     authored times (washes bloom in, ink draws tip to tail). This is the
//     default way to show a watercolor being painted. It keeps the sheet's
//     aspect ratio (size it with style, e.g. {position:'absolute', left,
//     top, width}). scale is the layers' render resolution over width x
//     height (default 1, a deliberate weight-over-dpi trade — raise it
//     toward the zoom factor if the composition zooms into the painting,
//     or toward the devicePixelRatio for a hero-sized sheet); quality is
//     0..1 layer image quality (default 0.92; 1 is the encoder's maximum).
//   useWatercolorLayers(fn, {width, height, seed, scale, quality}) -> L
//     (null if the kit isn't loaded — load watercolor_kit.js before the
//     engine — or if the painting fails to build) — the painting taken apart into strokes, for
//     choreography beyond in-order painting: L.count strokes, L.kind(i)
//     ('wash' | 'gradedWash' | 'glaze' | 'ink' | 'hatch' | 'splatter' |
//     'dryStroke' | 'reserve' | 'caption'), L.span(i) = the stroke's
//     {from, to} share of the painting's 0..1 timeline; call L.warm()
//     once after load so finished strokes pre-render off the critical
//     path (WatercolorPainting does this itself). Compose with:
//   <WatercolorSheet layers={L} style>children</WatercolorSheet> — the
//     paper the strokes sit on (keeps the sheet's aspect ratio), and
//   <WatercolorStroke index={i} at={0..1} style /> — stroke i as its own
//     element, placed where it was painted; at is its painting progress
//     (0 hidden, 1 finished — drive it from T with animate()); style lets
//     you move, scale, rotate, or fade the stroke (transform / opacity).
//     Strokes are paint, so they multiply: overlapping strokes darken
//     where they cross, as in the still image, within a few 8-bit levels
//     (tighter still at quality 1). The sheet clips to its
//     box — for strokes that fly in from outside it, set
//     style={{overflow: 'visible'}} on the WatercolorSheet. 'reserve' strokes
//     are erasures (lifted paper) — keep them where they were painted and
//     reveal them in order after the strokes they erase; moving an erase
//     around has no sensible meaning.
//   <WatercolorReveal painting={fn} from={CUES.X} to={CUES.Y} width height
//     seed steps scale format quality style />, or <WatercolorReveal
//     frames={[src, ...]} from to /> — the whole painting as ONE flat
//     image that paints on (frames pre-baked in the background, so it is
//     the lightest option and the one to zoom or pan over as a single
//     picture). Prefer WatercolorPainting when the strokes themselves
//     should appear one by one or be individually animated. format is
//     the image MIME type (default image/jpeg), quality 0..1 (default
//     0.88). Frames bake at width x height times scale (default: the
//     device pixel ratio, capped at 2) — if the composition zooms INTO
//     the painting, raise scale toward the maximum zoom so frames stay
//     crisp. The kit caps a sheet at ~12M pixels and the components clamp
//     scale to stay under it; exported video sharpness also depends on
//     the export dialog's own resolution choice.
//   Motion: Easing.{linear, easeIn|Out|InOutQuad/Cubic/Quart/Expo/Sine,
//     easeIn|Out|InOutBack, easeOutElastic}, interpolate(input, output, ease),
//     animate({from, to, start, end, ease}) -> fn(T), clamp(v, min, max)
//   Plumbing (rarely needed): Stage, PlaybackBar, TimelineContext,
//     useTime, useTimeline
//   Seek event (host/export transport): 'data-om-seek-to-time-frame',
//     detail {time, sync, playing} — the stage owns it; never implement it
//     yourself
//
// THE AUTHORING CONTRACT — this is what makes the host timeline's trim and
// speed gestures write back into YOUR file, so follow it exactly:
//   1. Declare the scene list as a JSON string literal in a plain inline
//      <script> of the main document (NOT type="text/babel", NOT a sibling
//      .jsx — only vanilla inline scripts are addressable for write-back):
//        <script>window.OM_SCENES = '[{"name":"Opening","dur":3,"desc":"The logo fades in and the title settles"},{"name":"Build","dur":5,"desc":"Bars grow to their final values"}]';</script>
//      Give every entry a "desc": one short plain-words sentence saying
//      what happens in that section. The user reads it in the timeline's
//      section popover — keep it true whenever you edit the section.
//   2. Pass the string through untouched:
//        <CompositionStage scenes={window.OM_SCENES} ...>
//   3. ALSO declare the playback setting the same way:
//        <script>window.OM_PLAYBACK = '{"mode":"loop"}';</script>
//      and pass it through untouched (values: '{"mode":"loop"}' or
//      '{"mode":"times","count":N}'; omitting keeps loop behavior but
//      leaves the host Repeat control read-only for this document).
//   IMPORTANT — the exportable-video contract: CompositionStage/Stage OWNS
//   it (the data-om-exportable-video-with-duration-secs attribute, the
//   data-om-seek-to-time-frame listener, the svg/foreignObject wrapper,
//   and font inlining). NEVER put the exportable attribute on any other
//   element — a second "exportable root" makes the host timeline and the
//   video exporter bind to the wrong element, and playback control /
//   export silently break.
//
// HOW TIME WORKS: each OM_SCENES entry is a named slice of the authored
// timeline. CUES.Name is that section's authored start (the running sum of
// authored lengths, in literal order). useComposition().T is the authored
// clock: when the user trims or speeds a section on the host timeline, the
// engine replays that section's SAME authored slice over the new playback
// length — your choreography retimes, never cuts off. The optional "nat"
// field on an entry is the engine's authored-length anchor — the host
// timeline stamps it on the first retime; don't set it by hand.
//
// CUE-FIRST DISCIPLINE (what makes a piece read as one continuous video):
//   1. Write the OM_SCENES literal FIRST — it is the piece's outline.
//   2. One helper component per section for readability, but ALL of them
//      render ALL the time inside the one tree, keyed to CUES — never
//      conditionally mounted per section.
//   3. Define exactly three motion helpers up front (e.g.
//      MOTION = {enter, draw, pop} wrapping Easing curves) and use no
//      easing or transform outside them; one caption element, one visible
//      at a time (<Captions> has this built in).
//   A shared element that crosses a boundary is just motion whose start
//   and end straddle a cue: animate({from, to, start: CUES.Build - 0.4,
//   end: CUES.Build + 0.6})(T) glides through the boundary, and a user
//   slowing either section slows the glide without breaking it.
//
// RENDER FROM T ONLY: the exporter seeks each frame with a synchronous
// commit and may serialize the stage the moment the seek event returns —
// anything painted from useEffect or your own requestAnimationFrame lags
// that commit and exports stale. Render everything visible from T and this
// is automatic. A seeked frame is a deterministic render at that time.
//
// HARD CUTS are content now, not structure: wrap a shot's elements in
// <Shot from to> (visibility toggles at the cues; children stay mounted so
// images and videos hold their readiness). Shot also doubles as the
// perf gate for heavy far-away beats.
//
// LOOP SEAMS are the one surviving boundary rule: a looping piece shows
// its last authored frame immediately before its first — make them match
// (settle your choreography by authoredTotal, open it at 0).
//
// DIAGNOSTICS: choreography that references an unknown section name (a
// rename or deletion in OM_SCENES) shows a badge below the stage in the
// preview, outside the exportable svg — visible in preview screenshots,
// never in the exported video. An OM_SCENES section with no choreography
// keyed to it is a valid empty beat, not an error.
/* END USAGE */

// ─────────────────────────────────────────────────────────────────────────────

// ── Easing functions (hand-rolled, Popmotion-style) ─────────────────────────
// All easings take t ∈ [0,1] and return eased t ∈ [0,1] (may overshoot for back/elastic).
const Easing = {
  linear: t => t,
  // Quad
  easeInQuad: t => t * t,
  easeOutQuad: t => t * (2 - t),
  easeInOutQuad: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  // Cubic
  easeInCubic: t => t * t * t,
  easeOutCubic: t => --t * t * t + 1,
  easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
  // Quart
  easeInQuart: t => t * t * t * t,
  easeOutQuart: t => 1 - --t * t * t * t,
  easeInOutQuart: t => t < 0.5 ? 8 * t * t * t * t : 1 - 8 * --t * t * t * t,
  // Expo
  easeInExpo: t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  easeOutExpo: t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
  easeInOutExpo: t => {
    if (t === 0) return 0;
    if (t === 1) return 1;
    if (t < 0.5) return 0.5 * Math.pow(2, 20 * t - 10);
    return 1 - 0.5 * Math.pow(2, -20 * t + 10);
  },
  // Sine
  easeInSine: t => 1 - Math.cos(t * Math.PI / 2),
  easeOutSine: t => Math.sin(t * Math.PI / 2),
  easeInOutSine: t => -(Math.cos(Math.PI * t) - 1) / 2,
  // Back (overshoot)
  easeOutBack: t => {
    const c1 = 1.70158,
      c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
  easeInBack: t => {
    const c1 = 1.70158,
      c3 = c1 + 1;
    return c3 * t * t * t - c1 * t * t;
  },
  easeInOutBack: t => {
    const c1 = 1.70158,
      c2 = c1 * 1.525;
    return t < 0.5 ? Math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2) / 2 : (Math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2;
  },
  // Elastic
  easeOutElastic: t => {
    const c4 = 2 * Math.PI / 3;
    if (t === 0) return 0;
    if (t === 1) return 1;
    return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  }
};

// ── Core interpolation helpers ──────────────────────────────────────────────

// Clamp a value to [min, max]
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

// interpolate([0, 0.5, 1], [0, 100, 50], ease?) -> fn(t)
// Popmotion-style: linearly maps t across input keyframes to output values,
// with optional easing per segment (single fn or array of fns).
function interpolate(input, output, ease = Easing.linear) {
  return t => {
    if (t <= input[0]) return output[0];
    if (t >= input[input.length - 1]) return output[output.length - 1];
    for (let i = 0; i < input.length - 1; i++) {
      if (t >= input[i] && t <= input[i + 1]) {
        const span = input[i + 1] - input[i];
        const local = span === 0 ? 0 : (t - input[i]) / span;
        const easeFn = Array.isArray(ease) ? ease[i] || Easing.linear : ease;
        const eased = easeFn(local);
        return output[i] + (output[i + 1] - output[i]) * eased;
      }
    }
    return output[output.length - 1];
  };
}

// animate({from, to, start, end, ease})(t) — simpler single-segment tween.
// Returns `from` before `start`, `to` after `end`.
function animate({
  from = 0,
  to = 1,
  start = 0,
  end = 1,
  ease = Easing.easeInOutCubic
}) {
  return t => {
    if (t <= start) return from;
    if (t >= end) return to;
    const local = (t - start) / (end - start);
    return from + (to - from) * ease(local);
  };
}

// ── Timeline context ────────────────────────────────────────────────────────

const TimelineContext = React.createContext({
  time: 0,
  duration: 10,
  playing: false
});
const useTime = () => React.useContext(TimelineContext).time;
const useTimeline = () => React.useContext(TimelineContext);

// How long a marked (detail.playing === true) host seek keeps the
// external-playback latch alive with no successor. The host play bar's
// seek pump is one-in-flight/latest-wins, so its inter-seek gap is tens
// of milliseconds in the worst case — 400ms is far above that, so a
// marked stream that dies mid-play decays the latch promptly.
var SS_EXT_PLAY_MS = 400;

// ── Font inlining ───────────────────────────────────────────────────────────
// Copy every @font-face rule from the page into a <style> inside the svg's
// foreignObject, with font URLs rewritten to data: URLs. Makes the svg
// self-describing so serializing it alone (video export fast path) still
// renders with the right fonts. Sets data-om-fonts-inlined on the svg when
// done so the exporter can wait for it.

function useInlineFontsInto(svgRef) {
  React.useEffect(() => {
    const svg = svgRef.current;
    const host = svg && svg.querySelector('foreignObject > div');
    if (!svg || !host) return;
    let cancelled = false;
    (async () => {
      const rules = [];
      for (const ss of document.styleSheets) {
        let cssRules;
        try {
          cssRules = ss.cssRules;
        } catch {
          // Cross-origin sheet without crossorigin attr (e.g. the standard
          // fonts.googleapis.com <link>) — fetch the CSS text directly and
          // regex-extract the @font-face blocks.
          if (ss.href) {
            try {
              const txt = await fetch(ss.href).then(r => {
                if (!r.ok) throw 0;
                return r.text();
              });
              for (const ff of txt.match(/@font-face\s*{[^}]*}/g) || []) rules.push({
                css: ff,
                base: ss.href
              });
            } catch {}
          }
          continue;
        }
        if (!cssRules) continue;
        for (const r of cssRules) {
          if (r.type === CSSRule.FONT_FACE_RULE) {
            rules.push({
              css: r.cssText,
              base: ss.href || location.href
            });
          }
        }
      }
      const toDataURL = url => fetch(url).then(r => {
        if (!r.ok) throw 0;
        return r.blob();
      }).then(b => new Promise(res => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.onerror = () => res(url);
        fr.readAsDataURL(b);
      })).catch(() => url);
      const parts = await Promise.all(rules.map(async ({
        css,
        base
      }) => {
        const re = /url\((['"]?)([^'")]+)\1\)/g;
        let out = css,
          m;
        while (m = re.exec(css)) {
          const u = m[2];
          if (u.startsWith('data:')) continue;
          let abs;
          try {
            abs = new URL(u, base).href;
          } catch {
            continue;
          }
          out = out.split(m[0]).join(`url("${await toDataURL(abs)}")`);
        }
        return out;
      }));
      if (cancelled || !parts.length) {
        svg.setAttribute('data-om-fonts-inlined', 'true');
        return;
      }
      const style = document.createElement('style');
      style.textContent = parts.join('\n');
      host.insertBefore(style, host.firstChild);
      svg.setAttribute('data-om-fonts-inlined', 'true');
    })();
    return () => {
      cancelled = true;
    };
  }, []);
}
function Stage({
  width = 1280,
  height = 720,
  duration = 10,
  background = '#f6f4ef',
  fps = 60,
  loop = true,
  autoplay = true,
  // Parsed playback object ({mode:'loop'} | {mode:'times',count:N}) or
  // null. When present it overrides the legacy loop prop — CompositionStage
  // passes the validated value from the OM_PLAYBACK authoring contract.
  playback = null,
  persistKey = 'animstage-v3',
  children
}) {
  // Props arrive as strings when Stage is mounted via <x-import> (DC
  // projects) — coerce so style={{width}} gets a number React can px-ify.
  width = +width || 1280;
  height = +height || 720;
  duration = +duration || 10;
  fps = +fps || 60;
  if (typeof loop === 'string') loop = loop !== 'false';
  if (typeof autoplay === 'string') autoplay = autoplay !== 'false';
  const playTimes = playback && playback.mode === 'times' ? playback.count : null;
  const loopEff = playback ? playback.mode === 'loop' : loop;
  const [time, setTime] = React.useState(() => {
    try {
      const v = parseFloat(localStorage.getItem(persistKey + ':t') || '0');
      return isFinite(v) ? clamp(v, 0, duration) : 0;
    } catch {
      return 0;
    }
  });
  const [playing, setPlaying] = React.useState(autoplay);
  // The external-playback latch: true while the HOST play bar is driving
  // time forward as genuine continuous playback (its play-loop seeks
  // carry detail.playing === true). The engine's own clock stays paused
  // the whole time — exactly one clock ever drives — so this is a
  // separate bit, not a second meaning for `playing`. Set and cleared
  // in the seek handler below; decays via SS_EXT_PLAY_MS when the
  // marked stream stops without a parting unmarked seek.
  const [extPlay, setExtPlay] = React.useState(false);
  const extPlayTimerRef = React.useRef(null);
  const [hoverTime, setHoverTime] = React.useState(null);
  const [scale, setScale] = React.useState(1);
  const stageRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const rafRef = React.useRef(null);
  const lastTsRef = React.useRef(null);

  // Persist playhead
  React.useEffect(() => {
    try {
      localStorage.setItem(persistKey + ':t', String(time));
    } catch {}
  }, [time, persistKey]);

  // Auto-scale to fit viewport
  React.useEffect(() => {
    if (!stageRef.current) return;
    const el = stageRef.current;
    const measure = () => {
      const barH = 44; // playback bar height
      const s = Math.min(el.clientWidth / width, (el.clientHeight - barH) / height);
      setScale(Math.max(0.05, s));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [width, height]);

  // Passes completed since playback last started. Lives in a ref so the
  // per-frame wrap can count without re-running this effect; reset on
  // every (re)start so a fresh play (or a host restart) gets the full
  // run count again.
  const passesRef = React.useRef(0);

  // Animation loop
  React.useEffect(() => {
    if (!playing) {
      lastTsRef.current = null;
      return;
    }
    passesRef.current = 0;
    const step = ts => {
      if (lastTsRef.current == null) lastTsRef.current = ts;
      const dt = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;
      setTime(t => {
        let next = t + dt;
        if (next >= duration) {
          if (playTimes !== null) {
            // Play N times then hold the last frame — the partial pass a
            // mid-timeline start produces counts as a pass, so the piece
            // never runs longer than N full durations.
            passesRef.current += 1;
            if (passesRef.current >= playTimes) {
              next = duration;
              setPlaying(false);
            } else {
              next = next % duration;
            }
          } else if (loopEff) {
            next = next % duration;
          } else {
            next = duration;
            setPlaying(false);
          }
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTsRef.current = null;
    };
  }, [playing, duration, loopEff, playTimes]);

  // Keyboard: space = play/pause, ← → = seek
  React.useEffect(() => {
    const onKey = e => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        setPlaying(p => !p);
      } else if (e.code === 'ArrowLeft') {
        setTime(t => clamp(t - (e.shiftKey ? 1 : 0.1), 0, duration));
      } else if (e.code === 'ArrowRight') {
        setTime(t => clamp(t + (e.shiftKey ? 1 : 0.1), 0, duration));
      } else if (e.key === '0' || e.code === 'Home') {
        setTime(0);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [duration]);

  // Video-export protocol + the editor's play bar: hosts dispatch this
  // event per frame; pause + sync the playhead so the frame shows exactly
  // that timestamp. The host play bar marks its play-loop seeks with
  // detail.playing === true — the mark latches extPlay (playback is
  // playback even when a host clock drives it), while ANY unmarked seek
  // (scrub, step, export frame, the transport's pause park) clears the
  // latch in the same commit it retimes, so a seeked frame still renders
  // exactly one scene's state. The engine's own clock pauses either way.
  React.useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    // Sync-seek capability: a dispatcher that marks its seek with
    // detail.sync === true gets the commit applied via ReactDOM.flushSync,
    // so the stage DOM reflects the seeked frame the moment dispatchEvent
    // returns. The video exporter keys off the data-om-sync-seek
    // advertisement to drop its two-display-refresh settle (that wait only
    // exists to let React's async commit land — serialization needs the
    // committed DOM, not the paint). Feature-detected: a runtime without
    // ReactDOM.flushSync never advertises and every seek takes the async
    // path. Unmarked seeks (scrubs, the host play bar) stay async — a
    // forced sync render per pointermove would tax the editor for no one.
    const canSyncSeek = typeof ReactDOM !== 'undefined' && typeof ReactDOM.flushSync === 'function';
    const onSeek = e => {
      const apply = () => {
        setPlaying(false);
        const hostPlay = !!(e.detail && e.detail.playing === true);
        if (extPlayTimerRef.current) {
          clearTimeout(extPlayTimerRef.current);
          extPlayTimerRef.current = null;
        }
        if (hostPlay) {
          // Watchdog: the latch is only as alive as its seek stream. If the
          // host stops without a parting seek (tab jank, bar unmount), the
          // latch decays on its own rather than stranding extPlaying true.
          extPlayTimerRef.current = setTimeout(() => {
            extPlayTimerRef.current = null;
            setExtPlay(false);
          }, SS_EXT_PLAY_MS);
        }
        setExtPlay(hostPlay);
        setTime(clamp(e.detail.time, 0, duration));
      };
      // flushSync is safe here: a native DOM listener runs outside React's
      // lifecycle, and the exporter's dispatchEvent is synchronous, so the
      // commit lands in the same JS task — the engine's own rAF loop can
      // never interleave between seek and serialize.
      if (canSyncSeek && e.detail && e.detail.sync === true) {
        ReactDOM.flushSync(apply);
      } else {
        apply();
      }
    };
    el.addEventListener('data-om-seek-to-time-frame', onSeek);
    if (canSyncSeek) el.setAttribute('data-om-sync-seek', 'true');
    return () => {
      el.removeEventListener('data-om-seek-to-time-frame', onSeek);
      el.removeAttribute('data-om-sync-seek');
      if (extPlayTimerRef.current) {
        clearTimeout(extPlayTimerRef.current);
        extPlayTimerRef.current = null;
      }
      // Drop the latch too: this cleanup runs on every duration change
      // (an agent edit can retime mid-host-play, no gesture involved) and
      // the new effect instance arms no watchdog — clearing only the
      // timer could strand extPlay true forever if the marked stream died
      // in the gap. Fail toward cut: the next marked seek re-latches.
      setExtPlay(false);
    };
  }, [duration]);

  // Inline @font-face rules into the svg's foreignObject so the svg is
  // self-describing — serializing it alone (for video export) then renders
  // with the right fonts. Sets data-om-fonts-inlined once done.
  useInlineFontsInto(canvasRef);
  const displayTime = hoverTime != null ? hoverTime : time;
  const ctxValue = React.useMemo(
  // extPlaying is ADDITIVE: "time is advancing under an external
  // driver's continuous playback". `playing` keeps meaning the
  // engine's OWN clock — the hidden PlaybackBar glyph (and through it
  // the host's clock-reporter/adoption channel) reads that — and
  // CompositionClock is the one consumer that widens to either.
  () => ({
    time: displayTime,
    duration,
    playing,
    extPlaying: extPlay,
    setTime,
    setPlaying
  }), [displayTime, duration, playing, extPlay]);
  return (
    /*#__PURE__*/
    // data-om-starter: inert presence marker — Claude Design's starter-usage
    // probe reads it; it renders nothing. Keep it on this root element.
    React.createElement("div", {
      ref: stageRef,
      "data-om-starter": "animations-v3",
      style: {
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        background: '#0a0a0a',
        fontFamily: 'Inter, system-ui, sans-serif'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        flex: 1,
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        minHeight: 0
      }
    }, /*#__PURE__*/React.createElement("svg", {
      ref: canvasRef,
      width: width,
      height: height,
      "data-om-exportable-video-with-duration-secs": duration,
      style: {
        transform: `scale(${scale})`,
        transformOrigin: 'center',
        flexShrink: 0,
        boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
        display: 'block'
      }
    }, /*#__PURE__*/React.createElement("foreignObject", {
      x: "0",
      y: "0",
      width: "100%",
      height: "100%"
    }, /*#__PURE__*/React.createElement("div", {
      xmlns: "http://www.w3.org/1999/xhtml",
      style: {
        width,
        height,
        background,
        position: 'relative',
        overflow: 'hidden'
      }
    }, /*#__PURE__*/React.createElement(TimelineContext.Provider, {
      value: ctxValue
    }, children))))), /*#__PURE__*/React.createElement(PlaybackBar, {
      time: displayTime,
      actualTime: time,
      duration: duration,
      playing: playing,
      onPlayPause: () => setPlaying(p => !p),
      onReset: () => {
        setTime(0);
      },
      onSeek: t => setTime(t),
      onHover: t => setHoverTime(t)
    }))
  );
}

// ── Playback bar ────────────────────────────────────────────────────────────
// Play/pause, return-to-begin, scrub track, time display.
// Uses fixed-width time fields so layout doesn't thrash.

function PlaybackBar({
  time,
  duration,
  playing,
  onPlayPause,
  onReset,
  onSeek,
  onHover
}) {
  const trackRef = React.useRef(null);
  const [dragging, setDragging] = React.useState(false);
  const timeFromEvent = React.useCallback(e => {
    const rect = trackRef.current.getBoundingClientRect();
    const x = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    return x * duration;
  }, [duration]);
  const onTrackMove = e => {
    if (!trackRef.current) return;
    const t = timeFromEvent(e);
    if (dragging) {
      onSeek(t);
    } else {
      onHover(t);
    }
  };
  const onTrackLeave = () => {
    if (!dragging) onHover(null);
  };
  const onTrackDown = e => {
    setDragging(true);
    const t = timeFromEvent(e);
    onSeek(t);
    onHover(null);
  };
  React.useEffect(() => {
    if (!dragging) return;
    const onUp = () => setDragging(false);
    const onMove = e => {
      if (!trackRef.current) return;
      const t = timeFromEvent(e);
      onSeek(t);
    };
    window.addEventListener('mouseup', onUp);
    window.addEventListener('mousemove', onMove);
    return () => {
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('mousemove', onMove);
    };
  }, [dragging, timeFromEvent, onSeek]);
  const pct = duration > 0 ? time / duration * 100 : 0;
  const fmt = t => {
    const total = Math.max(0, t);
    const m = Math.floor(total / 60);
    const s = Math.floor(total % 60);
    const cs = Math.floor(total * 100 % 100);
    return `${String(m).padStart(1, '0')}:${String(s).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
  };
  const mono = 'JetBrains Mono, ui-monospace, SFMono-Regular, monospace';
  return /*#__PURE__*/React.createElement("div", {
    "data-omelette-chrome": true,
    style: {
      // Slimmed to visually match the host editor bar's basic row (the
      // single-scrubber look): transport first, tighter metrics, quieter
      // chrome. Shown only outside the app — the host bar suppresses this
      // whenever it is present.
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '6px 12px',
      background: 'rgba(20,20,20,0.92)',
      borderTop: '1px solid rgba(255,255,255,0.08)',
      width: '100%',
      maxWidth: 680,
      alignSelf: 'center',
      borderRadius: 6,
      color: '#f6f4ef',
      fontFamily: 'Inter, system-ui, sans-serif',
      userSelect: 'none',
      flexShrink: 0
    }
  }, /*#__PURE__*/React.createElement(IconButton, {
    onClick: onPlayPause,
    title: "Play/pause (space)"
  }, playing ? /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("rect", {
    x: "3",
    y: "2",
    width: "3",
    height: "10",
    fill: "currentColor"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "8",
    y: "2",
    width: "3",
    height: "10",
    fill: "currentColor"
  })) : /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M3 2l9 5-9 5V2z",
    fill: "currentColor"
  }))), /*#__PURE__*/React.createElement(IconButton, {
    onClick: onReset,
    title: "Return to start (0)"
  }, /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M3 2v10M12 2L5 7l7 5V2z",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinejoin: "round",
    strokeLinecap: "round"
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: mono,
      fontSize: 12,
      fontVariantNumeric: 'tabular-nums',
      width: 64,
      textAlign: 'right',
      color: '#f6f4ef'
    }
  }, fmt(time)), /*#__PURE__*/React.createElement("div", {
    ref: trackRef,
    onMouseMove: onTrackMove,
    onMouseLeave: onTrackLeave,
    onMouseDown: onTrackDown,
    style: {
      flex: 1,
      height: 22,
      position: 'relative',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      height: 4,
      background: 'rgba(255,255,255,0.12)',
      borderRadius: 2
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      width: `${pct}%`,
      height: 4,
      background: 'oklch(72% 0.12 250)',
      borderRadius: 2
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: `${pct}%`,
      top: '50%',
      width: 12,
      height: 12,
      marginLeft: -6,
      marginTop: -6,
      background: '#fff',
      borderRadius: 6,
      boxShadow: '0 2px 4px rgba(0,0,0,0.4)'
    }
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: mono,
      fontSize: 12,
      fontVariantNumeric: 'tabular-nums',
      width: 64,
      textAlign: 'left',
      color: 'rgba(246,244,239,0.55)'
    }
  }, fmt(duration)), typeof VideoEncoder !== 'undefined' && /*#__PURE__*/React.createElement(IconButton, {
    title: "Export video",
    onClick: () => window.parent.postMessage({
      type: 'omelette:request-video-export'
    }, '*')
  }, /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M7 2v7m0 0L4 6m3 3l3-3M2 12h10",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }))));
}
function IconButton({
  children,
  onClick,
  title
}) {
  const [hover, setHover] = React.useState(false);
  return /*#__PURE__*/React.createElement("button", {
    onClick: onClick,
    title: title,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
      width: 24,
      height: 24,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: hover ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 5,
      color: '#f6f4ef',
      cursor: 'pointer',
      padding: 0,
      transition: 'background 120ms'
    }
  }, children);
}

// ── Scene-list plumbing ──────────────────────────────────────────────────
// Guest-side validation of a scene list (the engine's own inputs: the
// authored prop, and host-dispatched updates). Mirrors the host parser's
// shape rules and constants — keep in sync with parseTimelineScenes in
// apps/web/src/shared/timeline.ts (16KB raw cap, 50 entries, dur finite in
// (0, 300]); returns null on any violation.
function ssParse(raw) {
  if (typeof raw !== 'string' || !raw || raw.length > 16 * 1024) return null;
  var parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    return null;
  }
  if (!Array.isArray(parsed) || parsed.length === 0 || parsed.length > 50) return null;
  for (var i = 0; i < parsed.length; i++) {
    var s = parsed[i];
    if (typeof s !== 'object' || s === null) return null;
    if (typeof s.name !== 'string' || typeof s.dur !== 'number') return null;
    if (!isFinite(s.dur) || s.dur <= 0 || s.dur > 300) return null;
  }
  return parsed;
}

// Guest-side validation of the playback value — mirrors the host parser
// (shared/timeline.ts parseTimelinePlayback): {"mode":"loop"} or
// {"mode":"times","count":1..99}, strict all-or-nothing, null otherwise.
// Callers treat null as the loop default.
function ppParse(raw) {
  if (typeof raw !== 'string' || !raw || raw.length > 256) return null;
  var parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null;
  var keys = Object.keys(parsed);
  if (parsed.mode === 'loop') return keys.length === 1 ? {
    mode: 'loop'
  } : null;
  if (parsed.mode === 'times') {
    if (keys.length !== 2) return null;
    var c = parsed.count;
    if (typeof c !== 'number' || c !== Math.floor(c) || c < 1 || c > 99) return null;
    return {
      mode: 'times',
      count: c
    };
  }
  return null;
}

// Stamps the playback attribute VERBATIM from the authored raw string (the
// host's write-back anchors on that exact value) and listens for the
// host's post-write update event. Same shape as SceneSync; only rendered
// when the document authors a playback literal — an absent contract means
// the attribute stays absent and the document plays its default.
function PlaybackSync(props) {
  var ref = React.useRef(null);
  var raw = props.raw;
  var onUpdate = props.onUpdate;
  React.useEffect(function () {
    var el = ref.current;
    if (!el) return;
    var root = el.closest('[data-om-exportable-video-with-duration-secs]');
    if (!root) return;
    root.setAttribute('data-om-timeline-playback', raw);
    var onEvent = function (e) {
      var next = e && e.detail;
      if (ppParse(next)) onUpdate(next);
    };
    root.addEventListener('data-om-timeline-playback-update', onEvent);
    return function () {
      root.removeEventListener('data-om-timeline-playback-update', onEvent);
      root.removeAttribute('data-om-timeline-playback');
    };
  }, [raw, onUpdate]);
  return /*#__PURE__*/React.createElement("div", {
    ref: ref,
    style: {
      display: 'none'
    }
  });
}

// Renders inside the Stage (so it can reach the exportable root via
// closest()): stamps the scenes attribute VERBATIM from the current raw
// string — the host's write-back anchors on that exact value — and listens
// for the host's post-write update event.
function SceneSync(props) {
  var ref = React.useRef(null);
  var raw = props.raw;
  var onUpdate = props.onUpdate;
  React.useEffect(function () {
    var el = ref.current;
    if (!el) return;
    var root = el.closest('[data-om-exportable-video-with-duration-secs]');
    if (!root) return;
    root.setAttribute('data-om-timeline-scenes', raw);
    var onEvent = function (e) {
      var next = e && e.detail;
      // Ignore anything that doesn't validate — a bad update must not tear
      // down a working composition.
      if (ssParse(next)) onUpdate(next);
    };
    root.addEventListener('data-om-timeline-scenes-update', onEvent);
    return function () {
      root.removeEventListener('data-om-timeline-scenes-update', onEvent);
      root.removeAttribute('data-om-timeline-scenes');
    };
  }, [raw, onUpdate]);
  return /*#__PURE__*/React.createElement("div", {
    ref: ref,
    style: {
      display: 'none'
    }
  });
}

// ── Continuous composition ──────────────────────────────────────────────

var CompositionContext = React.createContext(null);
function useComposition() {
  var ctx = React.useContext(CompositionContext);
  if (!ctx) throw new Error('useComposition() must be called inside <CompositionStage>');
  return ctx;
}
function ccDerive(scenes) {
  var playStart = 0;
  var authStart = 0;
  var sections = [];
  var table = Object.create(null);
  for (var i = 0; i < scenes.length; i++) {
    var s = scenes[i];
    var nat = typeof s.nat === 'number' && isFinite(s.nat) && s.nat > 0 ? s.nat : s.dur;
    sections.push({
      name: s.name,
      playStart: playStart,
      dur: s.dur,
      authStart: authStart,
      nat: nat
    });
    if (!Object.prototype.hasOwnProperty.call(table, s.name)) {
      table[s.name] = Math.round(authStart * 1000) / 1000;
    }
    playStart += s.dur;
    authStart += nat;
  }
  return {
    sections: sections,
    table: table,
    total: Math.round(playStart * 1000) / 1000,
    authoredTotal: Math.round(authStart * 1000) / 1000
  };
}
function ccWarp(d, t) {
  var ss = d.sections;
  if (ss.length === 0) return 0;
  var idx = ss.length - 1;
  for (var i = 0; i < ss.length; i++) {
    if (t < ss[i].playStart + ss[i].dur) {
      idx = i;
      break;
    }
  }
  var s = ss[idx];
  var local = Math.min(Math.max(t - s.playStart, 0), s.dur);
  var T = s.authStart + (s.dur > 0 ? local * (s.nat / s.dur) : 0);
  return Math.min(T, d.authoredTotal);
}
var CC_META = Object.assign(Object.create(null), {
  toString: 1,
  toLocaleString: 1,
  valueOf: 1,
  toJSON: 1,
  then: 1,
  constructor: 1,
  hasOwnProperty: 1,
  isPrototypeOf: 1,
  propertyIsEnumerable: 1,
  default: 1
});
function ccCueProxy(table, unknownRef) {
  if (typeof Proxy !== 'function') return table;
  return new Proxy(table, {
    get: function (target, prop) {
      if (typeof prop !== 'string' || prop in target) return target[prop];
      if (CC_META[prop] || prop.indexOf('@@') === 0) return Object.prototype[prop];
      unknownRef.current[prop] = true;
      return NaN;
    }
  });
}
function CcUnknownWatch(props) {
  var tl = useTimeline();
  React.useEffect(function () {
    var next = Object.keys(props.unknownRef.current).sort().join(', ');
    if (next !== props.badge) props.setBadge(next);
  }, [tl.time]);
  return null;
}
function CompositionClock(props) {
  var tl = useTimeline();
  var d = props.derived;
  var T = ccWarp(d, tl.time);
  var value = React.useMemo(function () {
    return {
      T: T,
      CUES: props.cues,
      time: tl.time,
      duration: tl.duration,
      authoredTotal: d.authoredTotal,
      playing: tl.playing || tl.extPlaying === true
    };
  }, [T, props.cues, tl.time, tl.duration, d, tl.playing, tl.extPlaying]);
  return /*#__PURE__*/React.createElement(CompositionContext.Provider, {
    value: value
  }, props.children);
}
function Shot(props) {
  var c = useComposition();
  var from = +props.from;
  var to = props.to == null ? Infinity : +props.to;
  var on = isFinite(from) && c.T >= from && c.T < to;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      visibility: on ? 'visible' : 'hidden'
    }
  }, props.children);
}
var CAPTION_FADE = 0.18;
function Captions(props) {
  var c = useComposition();
  var t = c.T;
  var items = (props.items || []).filter(function (it) {
    return it && isFinite(+it.at);
  }).sort(function (a, b) {
    return a.at - b.at;
  });
  var active = null;
  var end = Infinity;
  for (var i = 0; i < items.length; i++) {
    if (t < items[i].at) break;
    active = items[i];
    end = typeof active.until === 'number' && isFinite(active.until) ? active.until : i + 1 < items.length ? items[i + 1].at : Infinity;
  }
  if (!active || t >= end) return null;
  var o = Math.min(1, (t - active.at) / CAPTION_FADE);
  if (isFinite(end)) o = Math.min(o, (end - t) / CAPTION_FADE);
  o = Math.max(0, Math.min(1, o));
  return /*#__PURE__*/React.createElement("div", {
    "data-om-caption": true,
    style: Object.assign({
      position: 'absolute',
      left: '8%',
      right: '8%',
      bottom: '7%',
      textAlign: 'center',
      opacity: o,
      pointerEvents: 'none',
      font: '500 30px Inter, system-ui, sans-serif',
      color: '#f6f4ef',
      textShadow: '0 1px 14px rgba(0,0,0,0.45)'
    }, props.style)
  }, active.text);
}
function CompositionStage(props) {
  var width = +props.width || 1280;
  var height = +props.height || 720;
  var bg = props.bg || '#0b0b0e';
  var autoplay = props.autoplay == null ? true : String(props.autoplay) !== 'false';
  var loop = props.loop == null ? true : String(props.loop) !== 'false';
  var state = React.useState(props.scenes);
  var raw = state[0];
  var setRaw = state[1];
  var scenes = React.useMemo(function () {
    return ssParse(raw);
  }, [raw]);
  var pstate = React.useState(props.playback);
  var praw = pstate[0];
  var setPraw = pstate[1];
  var pb = React.useMemo(function () {
    return ppParse(praw);
  }, [praw]);
  var unknownRef = React.useRef({});
  var badgeState = React.useState('');
  var badge = badgeState[0];
  var setBadge = badgeState[1];
  var derived = React.useMemo(function () {
    unknownRef.current = {};
    return scenes ? ccDerive(scenes) : null;
  }, [scenes]);
  var cues = React.useMemo(function () {
    return derived ? ccCueProxy(derived.table, unknownRef) : null;
  }, [derived]);
  React.useEffect(function () {
    var next = Object.keys(unknownRef.current).sort().join(', ');
    if (next !== badge) setBadge(next);
  });
  if (!scenes) {
    return /*#__PURE__*/React.createElement("div", {
      style: {
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0b0b0e',
        color: '#c96442',
        font: '500 16px Inter, system-ui, sans-serif',
        textAlign: 'center'
      }
    }, "animations-v3: the scenes prop isn't a valid JSON scene list", /*#__PURE__*/React.createElement("br", null), "(expected '[", '{', "\"name\":\"\u2026\",\"dur\":N", '}', ", \u2026]')");
  }
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Stage, {
    width: width,
    height: height,
    duration: derived.total,
    background: bg,
    autoplay: autoplay,
    loop: loop,
    playback: pb
  }, /*#__PURE__*/React.createElement(SceneSync, {
    raw: raw,
    onUpdate: setRaw
  }), typeof praw === 'string' && praw !== '' && /*#__PURE__*/React.createElement(PlaybackSync, {
    raw: praw,
    onUpdate: setPraw
  }), /*#__PURE__*/React.createElement(CompositionClock, {
    derived: derived,
    cues: cues
  }, props.children), /*#__PURE__*/React.createElement(CcUnknownWatch, {
    unknownRef: unknownRef,
    badge: badge,
    setBadge: setBadge
  })), badge !== '' &&
  /*#__PURE__*/
  // Sibling of Stage, outside the exportable <svg>: visible in the
  // preview (and its screenshots), never in the exported video.
  React.createElement("div", {
    "data-om-unknown-cues": true,
    style: {
      position: 'absolute',
      left: 12,
      bottom: 56,
      zIndex: 10,
      padding: '6px 10px',
      borderRadius: 6,
      background: 'rgba(0,0,0,0.72)',
      color: '#e8906a',
      font: '500 12px Inter, system-ui, sans-serif',
      pointerEvents: 'none'
    }
  }, "choreography references unknown section", badge.indexOf(',') >= 0 ? 's' : '', ": ", badge));
}

// Strokes as layers: paint multiplies, so stroke images stacked with
// mix-blend-mode:multiply over the paper reproduce the flat render.

var WC_PIXEL_CAP = 11000000;
function wcLayerOpts(props) {
  var w = +props.width || 900,
    h = +props.height || 1200;
  var askScale = +props.scale || 1;
  return {
    width: w,
    height: h,
    scale: Math.min(askScale, Math.sqrt(WC_PIXEL_CAP / (w * h))),
    seed: props.seed == null ? undefined : +props.seed,
    quality: props.quality == null ? undefined : +props.quality
  };
}
var wcWarned = {};
function wcWarnOnce(key, message, err) {
  if (wcWarned[key]) return;
  wcWarned[key] = true;
  console.warn(message, err);
}
function useWatercolorLayers(painting, opts) {
  var kit = window.WatercolorKit;
  if (typeof painting !== 'function' || !kit || typeof kit.layers !== 'function') return null;
  try {
    return kit.layers(painting, wcLayerOpts(opts || {}));
  } catch (e) {
    wcWarnOnce('layers:' + e, 'watercolor painting failed to build; rendering the fallback sheet', e);
    return null;
  }
}
var WatercolorSheetContext = React.createContext(null);
function WatercolorSheet(props) {
  var L = props.layers || null;
  var style = Object.assign({
    position: 'relative',
    display: 'block',
    width: '100%',
    aspectRatio: L ? L.width + ' / ' + L.height : '3 / 4',
    isolation: 'isolate',
    overflow: 'hidden'
  }, props.style);
  if (!L) {
    return /*#__PURE__*/React.createElement("div", {
      style: Object.assign(style, {
        background: '#f4f1e8',
        color: '#8a8270',
        font: '12px system-ui, sans-serif',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      })
    }, "watercolor-kit.js not loaded (or the painting failed to build)");
  }
  return /*#__PURE__*/React.createElement(WatercolorSheetContext.Provider, {
    value: L
  }, /*#__PURE__*/React.createElement("div", {
    style: style,
    "data-om-watercolor-sheet": true
  }, /*#__PURE__*/React.createElement("img", {
    src: L.paper,
    alt: props.alt || '',
    style: {
      position: 'absolute',
      left: 0,
      top: 0,
      width: '100%',
      height: '100%',
      display: 'block'
    }
  }), props.children));
}
function WatercolorStroke(props) {
  var fromSheet = React.useContext(WatercolorSheetContext);
  var L = props.layers || fromSheet;
  if (!L) return null;
  var i = +props.index;
  if (!(i >= 0) || i >= L.count) return null;
  var at = props.at == null ? 1 : clamp(+props.at, 0, 1);
  if (!(at > 0)) return null;
  var box, src;
  try {
    box = L.box(i);
    src = box ? L.src(i, at) : null;
  } catch (e) {
    wcWarnOnce('stroke:' + i + ':' + e, 'watercolor stroke ' + i + ' failed to render; skipping it', e);
    return null;
  }
  if (!box || !src) return null;
  var style = Object.assign({
    position: 'absolute',
    display: 'block',
    left: box.x * 100 + '%',
    top: box.y * 100 + '%',
    width: box.w * 100 + '%',
    height: box.h * 100 + '%',
    mixBlendMode: L.kind(i) === 'reserve' ? 'normal' : 'multiply',
    pointerEvents: 'none'
  }, props.style);
  return /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: "",
    "data-om-watercolor-stroke": i,
    "data-om-stroke-kind": L.kind(i),
    style: style
  });
}

// The default watercolor moment: the painting assembled from its strokes,
// each appearing in painting order (a pure function of T).
function WatercolorPainting(props) {
  var c = useComposition();
  var from = +props.from || 0;
  var to = props.to == null ? from + 6 : +props.to;
  var u = clamp((c.T - from) / Math.max(to - from, 0.001), 0, 1);
  var eased = Easing.easeInOutQuad(u);
  var L = useWatercolorLayers(props.painting, props);
  var tick = React.useState(0)[1];
  var warmed = React.useRef(null);
  React.useEffect(function () {
    if (!L || typeof L.warm !== 'function') return;
    var p = L.warm();
    if (warmed.current === p) return;
    var live = true;
    p.then(function () {
      warmed.current = p;
      if (live) tick(function (x) {
        return x + 1;
      });
    });
    return function () {
      live = false;
    };
  }, [L && L.paper, props.painting]);
  var strokes = [];
  if (L) {
    for (var i = 0; i < L.count; i++) {
      var sp = L.span(i);
      var at = clamp((eased - sp.from) / Math.max(sp.to - sp.from, 1e-6), 0, 1);
      if (at <= 0) break;
      strokes.push(/*#__PURE__*/React.createElement(WatercolorStroke, {
        key: i,
        layers: L,
        index: i,
        at: at
      }));
    }
  }
  return /*#__PURE__*/React.createElement(WatercolorSheet, {
    layers: L,
    style: props.style,
    alt: props.alt
  }, strokes);
}

// Paint-on watercolor reveal as a pure function of T — an <img> with a data:
// URL (the exporter serializes those as-is; a live canvas would export blank).
function WatercolorReveal(props) {
  var c = useComposition();
  var from = +props.from || 0;
  var to = props.to == null ? from + 6 : +props.to;
  var u = clamp((c.T - from) / Math.max(to - from, 0.001), 0, 1);
  var style = Object.assign({
    display: 'block',
    width: '100%',
    height: '100%',
    objectFit: 'contain'
  }, props.style);
  var frames = Array.isArray(props.frames) && props.frames.length ? props.frames : null;
  var steps = frames ? frames.length - 1 : Math.max(1, Math.round(+props.steps || 36));
  var i = Math.min(steps, Math.round(Easing.easeInOutQuad(u) * steps));
  var painting = typeof props.painting === 'function' ? props.painting : null;
  var kit = window.WatercolorKit;
  var w = +props.width || 900,
    h = +props.height || 1200;
  var askScale = +props.scale || Math.min(2, window.devicePixelRatio || 1);
  var opts = {
    width: w,
    height: h,
    scale: Math.min(askScale, Math.sqrt(11000000 / (w * h))),
    seed: props.seed == null ? undefined : +props.seed,
    steps: steps,
    type: props.format || 'image/jpeg',
    quality: props.quality == null ? 0.88 : +props.quality
  };
  var key = opts.width + 'x' + opts.height + '#' + opts.seed + '@' + opts.scale + '/' + steps + ':' + opts.type + '/' + opts.quality;
  var cache = React.useRef({
    fn: null,
    key: '',
    frames: {},
    baking: false
  }).current;
  var tick = React.useState(0)[1];
  if (cache.fn !== painting && String(cache.fn) !== String(painting) || cache.key !== key) {
    cache.key = key;
    cache.frames = {};
    cache.baking = false;
  }
  cache.fn = painting;
  React.useEffect(function () {
    if (frames || cache.baking || !painting || !kit || typeof kit.bake !== 'function') return;
    cache.baking = true;
    var target = cache.frames;
    try {
      kit.bake(painting, opts, function (n, _t, url) {
        target[n] = url;
      }).then(function (all) {
        if (cache.frames !== target) return;
        for (var n = 0; n < all.length; n++) target[n] = all[n];
        tick(function (x) {
          return x + 1;
        });
      }).catch(function () {
        /* failed bake: the guarded lazy path below still renders */
      });
    } catch (e) {
      /* oversized painting: the guarded lazy path below still renders */
    }
  });
  if (frames) return /*#__PURE__*/React.createElement("img", {
    src: frames[i],
    alt: props.alt || '',
    style: style
  });
  if (!kit || !painting) {
    return /*#__PURE__*/React.createElement("div", {
      style: Object.assign({
        width: '100%',
        height: '100%',
        background: '#f4f1e8',
        color: '#8a8270',
        font: '12px system-ui, sans-serif',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }, props.style)
    }, "watercolor-kit.js not loaded (or no painting function)");
  }
  if (!cache.frames[i]) {
    try {
      cache.frames[i] = kit.frame(painting, Object.assign({}, opts, {
        at: i / steps
      }));
    } catch (e) {
      return /*#__PURE__*/React.createElement("div", {
        style: Object.assign({
          width: '100%',
          height: '100%',
          background: '#f4f1e8',
          color: '#8a8270',
          font: '12px system-ui, sans-serif',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }, props.style)
      }, "painting too large to render (", String(e && e.message).slice(0, 80), ")");
    }
  }
  return /*#__PURE__*/React.createElement("img", {
    src: cache.frames[i],
    alt: props.alt || '',
    style: style
  });
}
Object.assign(window, {
  Easing,
  interpolate,
  animate,
  clamp,
  TimelineContext,
  useTime,
  useTimeline,
  Stage,
  PlaybackBar,
  CompositionStage,
  useComposition,
  Shot,
  Captions,
  WatercolorReveal,
  WatercolorPainting,
  WatercolorSheet,
  WatercolorStroke,
  useWatercolorLayers
});
;
/* src/animations/pmhc-intro/pmhc-scene-v2.jsx */
// pMHC intro — continuous composition on animations-v3.
const W = 1920,
  H = 1080,
  S = 1.5; // shots are 1280×720 → stage ×1.5
const MOTION = {
  enter: Easing.easeOutCubic,
  draw: Easing.easeInOutCubic,
  pop: Easing.easeOutBack
};
const C = {
  ink: '#2B2D42',
  helix: '#8A8B90',
  crimson: '#B22222',
  pepA: '#F4D06F',
  pepB: '#C44D96',
  pepC: '#D0434F',
  pepD: '#B0609E',
  tcrA: '#00B4D8',
  tcrB: '#1D3557',
  glow: '#E0FAFF'
};
const FONT = '"IBM Plex Mono", ui-monospace, monospace';
const tw = (T, a, b, ease) => (ease || MOTION.draw)(clamp((T - a) / (b - a), 0, 1));
const lerp = (a, b, p) => a + (b - a) * p;
const win = (T, a, b, fi, fo) => Math.min(tw(T, a - fi, a + 0.0001, Easing.linear), 1 - tw(T, b, b + fo, Easing.linear));
const IMG = n => 'shots/' + n + '.png';
function Wash() {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)'
    }
  });
}

// One photographic layer: wash + multiplied render + unscaled overlay.
// cam = {s, ox, oy} in image px (1280 space). Overlays get map(x,y) → stage px.
function Frame({
  src,
  op,
  cam,
  clip,
  dx = 0,
  dy = 0,
  children,
  extra
}) {
  if (op <= 0.001) return null;
  const s = cam ? cam.s : 1,
    ox = (cam ? cam.ox : 640) * S,
    oy = (cam ? cam.oy : 360) * S;
  const map = (x, y) => [ox + (x * S + dx * S - ox) * s, oy + (y * S + dy * S - oy) * s];
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: op
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      mixBlendMode: 'multiply',
      transformOrigin: `${ox}px ${oy}px`,
      transform: `scale(${s})`
    }
  }, (Array.isArray(src) ? src : [src]).map((l, i) => /*#__PURE__*/React.createElement("img", {
    key: i,
    src: IMG(l.n || l),
    style: {
      position: 'absolute',
      left: (l.x || 0) * S,
      top: (l.y || 0) * S,
      width: W,
      height: H,
      opacity: l.o == null ? 1 : l.o,
      clipPath: l.clip || clip,
      transformOrigin: l.org ? `${l.org[0] * S}px ${l.org[1] * S}px` : undefined,
      transform: l.r ? `rotate(${l.r}deg)` : undefined
    }
  })), extra), children ? children(map) : null);
}

// Leader-line callout. anchors in image px; text pos in image px.
function Callout({
  map,
  p: p0,
  anchors,
  at,
  text,
  color,
  align = 'left',
  size = 24
}) {
  const p = clamp(p0 * 2, 0, 1);
  if (p <= 0.001) return null;
  const [tx, ty] = map(at[0], at[1]);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("svg", {
    width: W,
    height: H,
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'visible',
      pointerEvents: 'none'
    }
  }, anchors.map((a, i) => {
    const [ax, ay] = map(a[0], a[1]);
    const len = Math.hypot(ax - tx, ay - ty);
    return /*#__PURE__*/React.createElement("g", {
      key: i,
      opacity: Math.min(1, p * 1.5)
    }, /*#__PURE__*/React.createElement("line", {
      x1: tx,
      y1: ty,
      x2: ax,
      y2: ay,
      stroke: C.ink,
      strokeWidth: "2",
      strokeDasharray: len,
      strokeDashoffset: len * (1 - p)
    }), /*#__PURE__*/React.createElement("circle", {
      cx: ax,
      cy: ay,
      r: 7 * clamp(p * 2 - 1, 0, 1),
      fill: "#fff",
      stroke: C.ink,
      strokeWidth: "2.5"
    }));
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: tx,
      top: ty,
      transform: `translate(${align === 'right' ? '-100%' : align === 'center' ? '-50%' : '0'}, -50%) translateY(${(1 - p) * 8}px)`,
      opacity: clamp(p * 2 - 0.6, 0, 1),
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '6px 12px',
      background: 'rgba(255,255,255,0.92)',
      borderRadius: 4,
      border: '1.5px solid ' + C.ink,
      font: `500 ${size}px ${FONT}`,
      color: C.ink,
      whiteSpace: 'nowrap'
    }
  }, color ? /*#__PURE__*/React.createElement("span", {
    style: {
      width: 14,
      height: 14,
      borderRadius: 3,
      background: color,
      border: '1.5px solid ' + C.ink
    }
  }) : null, text));
}
function Chip({
  x,
  y,
  op,
  color,
  text,
  align = 'right'
}) {
  if (op <= 0.001) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      top: y,
      [align]: x,
      opacity: op,
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '8px 14px',
      background: 'rgba(255,255,255,0.92)',
      border: '1.5px solid ' + C.ink,
      borderRadius: 4,
      font: `500 24px ${FONT}`,
      color: C.ink,
      whiteSpace: 'nowrap'
    }
  }, color ? /*#__PURE__*/React.createElement("span", {
    style: {
      width: 18,
      height: 18,
      borderRadius: 3,
      background: color,
      border: '1.5px solid ' + C.ink
    }
  }) : null, text);
}

// ── Beat 1: HLA, groove, anatomy ──────────────────────────────────────────
const PEP = [[500, 398], [540, 372], [583, 356], [625, 362], [666, 350], [706, 358], [748, 366], [790, 348], [832, 362]];
function BeatHLA({
  T,
  A,
  L
}) {
  const fadeIn = tw(T, 0, 0.6, Easing.linear);
  const opA = fadeIn * (T < A.Groove + 0.6 ? 1 : 0);
  const camA = {
    s: lerp(1, 1.07, tw(T, 0, A.Groove + 0.6, Easing.linear)),
    ox: 640,
    oy: 360
  };
  const opB = tw(T, A.Groove - 0.1, A.Groove + 0.6, Easing.linear) * (T < A.Anatomy + 0.6 ? 1 : 0);
  const camB = {
    s: lerp(1.22, 1.42, tw(T, A.Groove - 0.1, A.Anatomy + 0.6, MOTION.enter)),
    ox: 630,
    oy: 330
  };
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Frame, {
    src: "shot1a_hla_surface",
    op: opA,
    cam: camA
  }, map => L && /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, 0.9, 1.7),
    anchors: [[780, 330]],
    at: [930, 250],
    text: "HLA class I"
  })), /*#__PURE__*/React.createElement(Frame, {
    src: "shot1b_groove_surface",
    op: opB,
    cam: camB
  }, map => L && /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Groove + 0.5, A.Groove + 1.3),
    anchors: [[700, 245]],
    at: [860, 170],
    text: "peptide-binding groove"
  })));
}

// Shared camera for every groove-view frame so the groove stays pixel-identical across cuts.
function grooveCam(T, A) {
  return {
    s: lerp(1.0, 1.1, tw(T, A.Anatomy - 0.1, A.Approach, Easing.linear)),
    ox: 660,
    oy: 400
  };
}
function BeatGroove({
  T,
  A,
  L
}) {
  const cam = grooveCam(T, A);
  const inA = tw(T, A.Anatomy - 0.1, A.Anatomy + 0.6, Easing.linear);
  const scaleIn = lerp(0.94, 1, tw(T, A.Anatomy - 0.1, A.Anatomy + 0.8, MOTION.enter));
  const camIn = {
    ...cam,
    s: cam.s * scaleIn
  };
  const d = 0.1; // ≈6-frame dissolve
  const step = (A.Lock - A.Candidates) / 4;
  const opC = inA * (T < A.Candidates ? 1 : 0);
  const out = 1 - tw(T, A.Approach - 0.15, A.Approach + 0.35, Easing.linear);
  const opG = (T >= A.Candidates ? 1 : 0) * out;
  const cands = [{
    n: 'pep_cand1',
    c: C.pepA
  }, {
    n: 'pep_cand2',
    c: C.pepB
  }, {
    n: 'pep_cand3',
    c: C.pepC
  }, {
    n: 'pep_cand4',
    c: C.pepD
  }];
  const candOp = i => {
    const a = A.Candidates + i * step,
      b = a + step;
    return tw(T, a - d / 2, a + d / 2, Easing.linear) * (T < b + d / 2 ? 1 : 0);
  };
  const opLock = tw(T, A.Lock - d / 2, A.Lock + d / 2, Easing.linear) * (T < A.Approach + 0.35 ? 1 : 0);
  const labOut = 1 - tw(T, A.Candidates - 0.4, A.Candidates - 0.05, Easing.linear);
  const ci = clamp(Math.floor((T - A.Candidates) / step), 0, 3);
  const inCand = T >= A.Candidates - d / 2 && T < A.Lock;
  const chipOp = tw(T, A.Candidates - 0.1, A.Candidates + 0.2, Easing.linear) * out;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Frame, {
    src: "shot1c_groove_cartoon",
    op: opC,
    cam: camIn
  }, map => L && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Anatomy + 0.9, A.Anatomy + 1.6) * labOut,
    anchors: [[380, 262]],
    at: [240, 175],
    text: "\u03B11 helix",
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Anatomy + 1.4, A.Anatomy + 2.1) * labOut,
    anchors: [[330, 540]],
    at: [200, 470],
    text: "\u03B12 helix",
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Anatomy + 1.9, A.Anatomy + 2.6) * labOut,
    anchors: [[690, 432]],
    at: [1000, 560],
    text: "\u03B2-sheet floor"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Anatomy + 2.4, A.Anatomy + 3.1) * labOut,
    anchors: [[870, 352]],
    at: [1060, 330],
    text: "peptide \xB7 9 aa",
    color: C.crimson
  }), PEP.map(([x, y], i) => {
    const p = tw(T, A.Anatomy + 3.0 + i * 0.06, A.Anatomy + 3.2 + i * 0.06, MOTION.pop) * labOut;
    if (p <= 0.001) return null;
    const [sx, sy] = map(x, y);
    const anchor = i === 1 || i === 8;
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        position: 'absolute',
        left: sx,
        top: sy - 52,
        transform: `translate(-50%,0) scale(${p})`,
        font: `600 20px ${FONT}`,
        color: anchor ? '#fff' : C.ink,
        background: anchor ? C.crimson : 'rgba(255,255,255,0.94)',
        border: '1.5px solid ' + C.ink,
        borderRadius: 3,
        padding: '1px 5px'
      }
    }, "P", i + 1);
  }))), /*#__PURE__*/React.createElement(Frame, {
    op: opG,
    cam: cam,
    src: [{
      n: 'layer_groove_empty'
    }, {
      n: T < A.Lock ? cands[ci].n : 'layer_peptide_alpha'
    }]
  }), L && inCand && /*#__PURE__*/React.createElement(Chip, {
    x: 80,
    y: 80,
    op: chipOp,
    color: cands[ci].c,
    text: `candidate ${ci + 1} / 4`
  }), L && /*#__PURE__*/React.createElement(Chip, {
    x: 80,
    y: 80,
    op: tw(T, A.Lock - 0.05, A.Lock + 0.25, Easing.linear) * out,
    color: C.crimson,
    text: "bound \xB7 stable fit"
  }));
}

// ── Beat 2: TCR arrives ──────────────────────────────────────────────────
function BeatTCR({
  T,
  A,
  L
}) {
  const opD = tw(T, A.Approach - 0.15, A.Approach + 0.35, Easing.linear) * (T < A.Interface + 0.4 ? 1 : 0);
  const slide = tw(T, A.Approach + 0.1, A.Approach + 2.2, MOTION.enter);
  const off = [29 * (1 - slide), -250 * (1 - slide)]; // along the TCR's own long axis
  const camD = {
    s: lerp(1.0, 1.04, tw(T, A.Approach, A.Interface + 0.4, Easing.linear)),
    ox: 640,
    oy: 380
  };
  const opE = tw(T, A.Interface - 0.2, A.Interface + 0.4, Easing.linear) * (T < A.Wobble + 0.25 ? 1 : 0);
  const camE = {
    s: lerp(1.0, 1.08, tw(T, A.Interface - 0.2, A.Wobble, Easing.linear)),
    ox: 620,
    oy: 380
  };
  const lab = a => tw(T, a, a + 0.7);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Frame, {
    op: opD,
    cam: camD,
    src: [{
      n: 'shot2d_tcr_surface_approach',
      clip: `inset(${387 * S}px 0 0 0)`
    }, {
      n: 'shot2d_tcr_surface_approach',
      clip: `inset(0 0 ${(720 - 387) * S}px 0)`,
      x: off[0],
      y: off[1],
      o: tw(T, A.Approach, A.Approach + 0.6, Easing.linear)
    }]
  }, map => L && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Approach + 2.0),
    anchors: [[560, 230]],
    at: [400, 160],
    text: "TCR \u03B1",
    color: C.tcrA,
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Approach + 2.2),
    anchors: [[720, 230]],
    at: [880, 160],
    text: "TCR \u03B2",
    color: C.tcrB
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Approach + 2.4),
    anchors: [[740, 560]],
    at: [880, 600],
    text: "peptide\u2013HLA",
    color: C.helix
  }))), /*#__PURE__*/React.createElement(Frame, {
    src: "shot2e_vdomain_ribbons",
    op: opE,
    cam: camE
  }, map => L && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Interface + 0.5),
    anchors: [[520, 240]],
    at: [360, 180],
    text: "V\u03B1",
    color: C.tcrA,
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Interface + 0.7),
    anchors: [[740, 240]],
    at: [890, 180],
    text: "V\u03B2",
    color: C.tcrB
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Interface + 1.0),
    anchors: [[500, 345], [585, 372], [662, 372]],
    at: [330, 420],
    text: "CDR1\xB72\xB73 loops",
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Interface + 1.3),
    anchors: [[722, 412]],
    at: [930, 440],
    text: "peptide",
    color: C.crimson
  }))));
}

// ── Beat 3: stability ────────────────────────────────────────────────────
function pepPose(t) {
  // t: seconds into Wobble section
  if (t < 1.6) {
    const env = Math.max(0, 1 - t / 1.5);
    const w = 2 * Math.PI * 6 * t;
    return {
      x: 10 * Math.sin(w) * env,
      y: 3 * Math.sin(w * 1.3 + 1) * env,
      r: 4 * Math.sin(w + 0.6) * env,
      o: 1
    };
  }
  if (t < 2.5) {
    const p = MOTION.draw(clamp((t - 1.7) / 0.8, 0, 1));
    return {
      x: 62 * p,
      y: -96 * p,
      r: 15 * p,
      o: 1 - p
    };
  }
  if (t < 3.0) return {
    x: 62,
    y: -96,
    r: 15,
    o: 0
  };
  const p = clamp((t - 3.0) / 1.0, 0, 1);
  const e = MOTION.pop(p);
  const settle = Math.max(0, 1 - (t - 3.6) / 0.8);
  const jit = t > 3.6 ? 3 * Math.sin(2 * Math.PI * 4 * (t - 3.6)) * settle : 0;
  return {
    x: 62 * (1 - e) + jit,
    y: -96 * (1 - e),
    r: 15 * (1 - e),
    o: MOTION.enter(clamp(p * 2.5, 0, 1))
  };
}
function BeatStability({
  T,
  A,
  L
}) {
  const t = T - A.Wobble;
  const opW = tw(T, A.Wobble - 0.15, A.Wobble + 0.25, Easing.linear) * (T < A.Anchors + 0.5 ? 1 : 0);
  const cam = {
    s: lerp(1.04, 1.1, tw(T, A.Wobble, A.Contact, Easing.linear)),
    ox: 660,
    oy: 400
  };
  const org = [665, 365];
  const pose = pepPose(t);
  const layers = [{
    n: 'layer_groove_empty'
  }];
  if (t > 1.7 && t < 2.6) {
    for (let k = 4; k >= 1; k--) {
      const g = pepPose(t - k * 0.05);
      layers.push({
        n: 'layer_peptide_alpha',
        x: g.x,
        y: g.y,
        r: g.r,
        org,
        o: g.o * 0.22 * (1 - k / 5)
      });
    }
  }
  layers.push({
    n: 'layer_peptide_alpha',
    x: pose.x,
    y: pose.y,
    r: pose.r,
    org,
    o: pose.o
  });
  const opAn = tw(T, A.Anchors, A.Anchors + 0.5, Easing.linear) * (T < A.Contact + 0.4 ? 1 : 0);
  const pk = a => tw(T, a, a + 0.8);
  const unst = tw(T, A.Wobble + 0.2, A.Wobble + 0.5, Easing.linear) * (1 - tw(T, A.Wobble + 2.6, A.Wobble + 2.9, Easing.linear));
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Frame, {
    src: layers,
    op: opW,
    cam: cam
  }), L && /*#__PURE__*/React.createElement(Chip, {
    x: 80,
    y: 80,
    op: unst * opW,
    color: C.crimson,
    text: t < 1.7 ? 'loose fit' : 'peptide lost · no signal'
  }), /*#__PURE__*/React.createElement(Frame, {
    src: "shot3d_settled_anchors_grey",
    op: opAn,
    cam: cam
  }, map => {
    const pocket = (x, y, p) => {
      const [cx, cy] = map(x, y);
      const rx = 58 * cam.s * S * 0.62,
        ry = 40 * cam.s * S * 0.62;
      const len = 2 * Math.PI * Math.sqrt((rx * rx + ry * ry) / 2);
      return /*#__PURE__*/React.createElement("ellipse", {
        cx: cx,
        cy: cy,
        rx: rx,
        ry: ry,
        fill: `rgba(178,34,34,${0.10 * p})`,
        stroke: C.crimson,
        strokeWidth: "3",
        strokeDasharray: `10 8`,
        strokeDashoffset: 0,
        opacity: p,
        style: {
          clipPath: 'none'
        },
        pathLength: len
      });
    };
    return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("svg", {
      width: W,
      height: H,
      style: {
        position: 'absolute',
        inset: 0
      }
    }, pocket(540, 374, pk(A.Anchors + 0.5)), pocket(832, 366, pk(A.Anchors + 0.8))), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: pk(A.Anchors + 0.9),
      anchors: [[540, 346]],
      at: [400, 175],
      text: "P2 \u2192 pocket B",
      color: C.crimson,
      align: "right"
    }), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: pk(A.Anchors + 1.2),
      anchors: [[832, 340]],
      at: [960, 175],
      text: "P9 \u2192 pocket F",
      color: C.crimson
    }));
  }));
}
function BeatContact({
  T,
  A,
  L
}) {
  const op = tw(T, A.Contact - 0.2, A.Contact + 0.4, Easing.linear) * (T < A.Activation + 0.3 ? 1 : 0);
  const cam = {
    s: lerp(1.0, 1.07, tw(T, A.Contact - 0.2, A.Activation, Easing.linear)),
    ox: 600,
    oy: 370
  };
  const fire = A.Contact + 1.4;
  return /*#__PURE__*/React.createElement(Frame, {
    src: "shot3e_contact_glow",
    op: op,
    cam: cam
  }, map => {
    const [gx, gy] = map(592, 378);
    const k = cam.s * S;
    const glow = tw(T, fire - 0.3, fire + 0.4, MOTION.enter) * (0.75 + 0.25 * Math.sin((T - fire) * 7));
    return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
      style: {
        position: 'absolute',
        left: gx - 150 * k,
        top: gy - 150 * k,
        width: 300 * k,
        height: 300 * k,
        borderRadius: '50%',
        background: `radial-gradient(circle, rgba(224,250,255,${0.85 * glow}) 0%, rgba(0,180,216,${0.18 * glow}) 45%, rgba(0,180,216,0) 70%)`,
        mixBlendMode: 'normal'
      }
    }), /*#__PURE__*/React.createElement("svg", {
      width: W,
      height: H,
      style: {
        position: 'absolute',
        inset: 0
      }
    }, [0, 1, 2].map(i => {
      const p = clamp((T - fire - i * 0.45) / 1.2, 0, 1);
      if (p <= 0 || p >= 1) return null;
      return /*#__PURE__*/React.createElement("circle", {
        key: i,
        cx: gx,
        cy: gy,
        r: (70 + 120 * MOTION.enter(p)) * k,
        fill: "none",
        stroke: C.tcrA,
        strokeWidth: 4 * (1 - p) + 1,
        opacity: 1 - p
      });
    })), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: tw(T, A.Contact + 0.5, A.Contact + 1.2),
      anchors: [[560, 352]],
      at: [250, 300],
      text: "CDR3\u03B1",
      color: C.tcrA,
      align: "right"
    }), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: tw(T, A.Contact + 0.7, A.Contact + 1.4),
      anchors: [[790, 298]],
      at: [990, 230],
      text: "CDR3\u03B2",
      color: C.tcrB
    }), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: tw(T, A.Contact + 0.9, A.Contact + 1.6),
      anchors: [[800, 372]],
      at: [1000, 400],
      text: "peptide",
      color: C.crimson
    }), L && /*#__PURE__*/React.createElement(Chip, {
      x: 80,
      y: 80,
      op: tw(T, fire, fire + 0.3, Easing.linear),
      color: C.tcrA,
      text: "signal"
    }));
  });
}

// ── Beat 4: response (vector) ────────────────────────────────────────────
function TCell({
  x,
  y,
  r,
  fill = C.tcrA,
  a = C.tcrA,
  b = C.tcrB,
  op = 1,
  glow = 0,
  s = 1
}) {
  if (op <= 0.001) return null;
  const R = r * s,
    cw = Math.max(3, r * 0.1),
    gap = r * 0.16,
    ch = r * 0.62;
  return /*#__PURE__*/React.createElement("g", {
    opacity: op
  }, glow > 0 && /*#__PURE__*/React.createElement("circle", {
    cx: x,
    cy: y,
    r: R * 1.45,
    fill: C.glow,
    opacity: glow
  }), /*#__PURE__*/React.createElement("rect", {
    x: x - gap - cw,
    y: y - R - ch + 2,
    width: cw,
    height: ch,
    rx: cw / 2,
    fill: a,
    stroke: C.ink,
    strokeWidth: Math.max(1.2, r * 0.016)
  }), /*#__PURE__*/React.createElement("rect", {
    x: x + gap,
    y: y - R - ch + 2,
    width: cw,
    height: ch,
    rx: cw / 2,
    fill: b,
    stroke: C.ink,
    strokeWidth: Math.max(1.2, r * 0.016)
  }), /*#__PURE__*/React.createElement("circle", {
    cx: x,
    cy: y,
    r: R,
    fill: fill,
    stroke: C.ink,
    strokeWidth: Math.max(2, r * 0.025)
  }));
}
const NAIVE = [[210, 330, '#F4D06F', '#C9A43A'], [330, 470, '#7FC8A9', '#2E7D5B'], [230, 620, '#E05A5A', '#8E1E1E'], [370, 760, '#7FB3E0', '#2D5E8E'], [250, 880, '#8E5AA8', '#4E2766'], [520, 300, '#C3A6D4', '#7C5C92'], [500, 560, '#2E6E43', '#173C24'], [520, 760, '#F2C6DC', '#B07893'], [700, 770, '#E68A2E', '#8C4A0E'], [700, 900, '#DCE8B8', '#8FA35C'], [620, 420, '#C44D96', '#7A2459']];
const CLONES = [[1130, 330], [1300, 450], [1480, 300], [1660, 420], [1170, 610], [1380, 680], [1590, 610], [1250, 860], [1500, 880], [1700, 780]];
const OTHERS = [[1440, 520, '#2E6E43', '#173C24'], [1760, 560, '#E68A2E', '#8C4A0E'], [1640, 230, '#F2C6DC', '#B07893'], [1120, 470, '#7FB3E0', '#2D5E8E']];
function BeatResponse({
  T,
  A,
  L
}) {
  const vis = tw(T, A.Activation - 0.2, A.Activation + 0.3, Easing.linear) * (T < A.Body + 0.25 ? 1 : 0);
  if (vis <= 0.001) return null;
  const t = T - A.Activation;
  const mv = tw(T, A.Expansion, A.Expansion + 1.1, MOTION.draw);
  const cx = lerp(960, 760, mv),
    cy = lerp(640, 560, mv),
    r = lerp(150, 46, mv);
  const sig = clamp((t - 0.4) / 0.7, 0, 1);
  const fired = clamp((t - 1.1) / 0.3, 0, 1);
  const pop = t > 1.1 ? 1 + 0.06 * Math.sin(clamp((t - 1.1) / 0.5, 0, 1) * Math.PI) : 1;
  const pm = (1 - tw(T, A.Expansion - 0.2, A.Expansion + 0.4, Easing.linear)) * tw(T, A.Activation, A.Activation + 0.5, MOTION.enter);
  const chTop = 640 - 150 - 93 + 2;
  const pmY = lerp(chTop - 120, chTop - 62, tw(T, A.Activation, A.Activation + 0.5, MOTION.enter));
  const clonesStart = A.Expansion + 1.2;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: vis
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement("svg", {
    width: W,
    height: H,
    style: {
      position: 'absolute',
      inset: 0
    }
  }, pm > 0.001 && /*#__PURE__*/React.createElement("g", {
    opacity: pm
  }, /*#__PURE__*/React.createElement("rect", {
    x: 960 - 90,
    y: pmY - 40,
    width: "180",
    height: "56",
    rx: "22",
    fill: "#BABBC0",
    stroke: C.ink,
    strokeWidth: "2.5"
  }), /*#__PURE__*/React.createElement("rect", {
    x: 960 - 46,
    y: pmY + 6,
    width: "92",
    height: "14",
    rx: "7",
    fill: C.crimson,
    stroke: C.ink,
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("rect", {
    x: 960 - 34,
    y: pmY - 120,
    width: "68",
    height: "80",
    rx: "8",
    fill: "#DCDDE2",
    stroke: C.ink,
    strokeWidth: "2.5"
  })), [0, 1, 2].map(i => {
    const p = clamp((t - 1.1 - i * 0.35) / 1.3, 0, 1);
    if (p <= 0 || p >= 1 || mv > 0.5) return null;
    return /*#__PURE__*/React.createElement("circle", {
      key: i,
      cx: cx,
      cy: cy,
      r: r * (1.05 + 0.9 * MOTION.enter(p)),
      fill: "none",
      stroke: C.tcrA,
      strokeWidth: 6 * (1 - p) + 1,
      opacity: (1 - p) * (1 - mv * 2)
    });
  }), NAIVE.map(([x, y, f, b], i) => {
    const p = tw(T, A.Expansion + 0.3 + i * 0.05, A.Expansion + 0.8 + i * 0.05, MOTION.pop);
    return /*#__PURE__*/React.createElement(TCell, {
      key: i,
      x: x,
      y: y,
      r: 38 * p,
      fill: f,
      a: f,
      b: b,
      op: clamp(p, 0, 1)
    });
  }), CLONES.map(([x, y], i) => {
    const a = clonesStart + i * 0.12;
    const lp = tw(T, a, a + 0.6, MOTION.draw);
    if (lp <= 0) return null;
    const ex = lerp(cx + r, x - 40, lp),
      ey = lerp(cy, y, lp);
    return /*#__PURE__*/React.createElement("line", {
      key: 'l' + i,
      x1: cx + r * 1.05,
      y1: cy,
      x2: ex,
      y2: ey,
      stroke: C.tcrA,
      strokeWidth: "3",
      opacity: "0.85"
    });
  }), OTHERS.map(([x, y, f, b], i) => {
    const p = tw(T, clonesStart + 0.6 + i * 0.1, clonesStart + 1.1 + i * 0.1, MOTION.pop);
    return /*#__PURE__*/React.createElement(TCell, {
      key: 'o' + i,
      x: x,
      y: y,
      r: 38 * p,
      fill: f,
      a: f,
      b: b,
      op: clamp(p, 0, 1)
    });
  }), CLONES.map(([x, y], i) => {
    const a = clonesStart + i * 0.12 + 0.45;
    const p = tw(T, a, a + 0.5, MOTION.pop);
    return /*#__PURE__*/React.createElement(TCell, {
      key: 'c' + i,
      x: x,
      y: y,
      r: 44 * p,
      op: clamp(p * 1.5, 0, 1)
    });
  }), /*#__PURE__*/React.createElement(TCell, {
    x: cx,
    y: cy,
    r: r,
    s: pop,
    glow: fired * 0.9 * (1 - mv)
  }), sig > 0 && sig < 1 && /*#__PURE__*/React.createElement("circle", {
    cx: 960 - 15 - 7,
    cy: lerp(chTop, 640 - 150, sig),
    r: "12",
    fill: C.glow,
    stroke: C.tcrA,
    strokeWidth: "3"
  })), L && /*#__PURE__*/React.createElement(Chip, {
    x: 80,
    y: 80,
    op: tw(T, A.Activation + 1.1, A.Activation + 1.4, Easing.linear) * (1 - mv),
    color: C.tcrA,
    text: "T cell \xB7 activated"
  }), L && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 460,
      top: 160,
      transform: 'translateX(-50%)',
      opacity: tw(T, A.Expansion + 0.6, A.Expansion + 1.2, Easing.linear),
      font: `500 30px ${FONT}`,
      color: C.ink
    }
  }, "na\xEFve repertoire"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 1420,
      top: 160,
      transform: 'translateX(-50%)',
      opacity: tw(T, clonesStart + 0.8, clonesStart + 1.4, Easing.linear),
      font: `500 30px ${FONT}`,
      color: C.ink
    }
  }, "effector & memory clones")));
}
function BeatBody({
  T,
  A
}) {
  const op = tw(T, A.Body - 0.15, A.Body + 0.25, Easing.linear);
  if (op <= 0.001) return null;
  const h = 860,
    w = 344 * h / 778;
  const s = lerp(1, 1.015, tw(T, A.Body, A.Body + 3, Easing.linear));
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: op
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement("img", {
    src: "shots/body_figure.png",
    style: {
      position: 'absolute',
      left: W / 2 - w / 2,
      top: 70,
      width: w,
      height: h,
      mixBlendMode: 'multiply',
      transform: `scale(${s})`,
      transformOrigin: '50% 60%'
    }
  }));
}
function ChapterTag({
  T,
  A
}) {
  const beats = [[0, '01', 'HLA presents a peptide'], [A.Candidates, '02', 'pairing'], [A.Wobble, '03', 'stability'], [A.Activation, '04', 'immune response']];
  let cur = beats[0];
  beats.forEach(b => {
    if (T >= b[0]) cur = b;
  });
  const since = T - cur[0];
  const op = clamp(since / 0.4, 0, 1) * tw(T, 0, 0.6, Easing.linear);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 80,
      top: 86,
      opacity: op,
      display: 'flex',
      gap: 16,
      alignItems: 'baseline',
      font: `500 22px ${FONT}`,
      color: C.ink,
      letterSpacing: '0.06em',
      textTransform: 'uppercase'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 600
    }
  }, cur[1]), /*#__PURE__*/React.createElement("span", {
    style: {
      width: 36,
      height: 2,
      background: C.ink,
      alignSelf: 'center'
    }
  }), /*#__PURE__*/React.createElement("span", null, cur[2]));
}
function Piece({
  labels,
  captions
}) {
  const {
    T,
    CUES: A
  } = useComposition();
  const L = labels !== false;
  const credit = 1 - tw(T, A.Activation - 0.2, A.Activation + 0.3, Easing.linear);
  return /*#__PURE__*/React.createElement("div", {
    "data-screen-label": `t=${Math.floor(T)}s`,
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      background: '#E9EBEF',
      fontFamily: FONT
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement(BeatHLA, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatGroove, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatTCR, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatStability, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatContact, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatResponse, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatBody, {
    T: T,
    A: A
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      pointerEvents: 'none',
      background: 'radial-gradient(ellipse 80% 75% at 50% 48%, rgba(43,45,66,0) 60%, rgba(43,45,66,0.14) 100%)'
    }
  }), /*#__PURE__*/React.createElement(ChapterTag, {
    T: T,
    A: A
  }), credit > 0.001 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      right: 80,
      bottom: 28,
      opacity: credit,
      font: `400 20px ${FONT}`,
      color: '#4A4C5E'
    }
  }, "PDB 2BNQ \xB7 1.70 \xC5"), captions !== false && /*#__PURE__*/React.createElement(Captions, {
    style: {
      font: `400 30px ${FONT}`,
      color: C.ink,
      textShadow: 'none',
      bottom: '6%',
      left: '6%',
      right: '6%'
    },
    items: [{
      at: 0.4,
      text: 'HLA class I presents short peptides on the cell surface.'
    }, {
      at: A.Anatomy + 0.4,
      text: 'The peptide lies in a groove: two α-helices over a β-sheet floor.'
    }, {
      at: A.Candidates + 0.2,
      text: 'Many candidate peptides. Which ones fit this HLA allele?'
    }, {
      at: A.Lock + 0.1,
      text: 'Only some form a precise pair.'
    }, {
      at: A.Approach + 0.3,
      text: 'A T-cell receptor docks onto the peptide–HLA complex.'
    }, {
      at: A.Interface + 0.2,
      text: 'Its CDR loops read the exposed peptide residues.'
    }, {
      at: A.Wobble + 0.3,
      text: 'A loose peptide escapes, and the T cell sees nothing.'
    }, {
      at: A.Anchors + 0.2,
      text: 'A stable fit: anchors P2 and P9 seat in pockets B and F.'
    }, {
      at: A.Contact + 0.3,
      text: 'CDR3 contacts the peptide. The receptor signals.'
    }, {
      at: A.Activation + 0.2,
      text: 'The T cell activates.'
    }, {
      at: A.Expansion + 0.3,
      text: 'Out of the naïve repertoire, one clone expands.'
    }, {
      at: A.Body + 0.2,
      text: 'Every successful immune response depends on this pairing.'
    }]
  }));
}
function PmhcVideo() {
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(CompositionStage, {
    width: W,
    height: H,
    scenes: window.OM_SCENES,
    playback: window.OM_PLAYBACK,
    bg: "#E9EBEF"
  }, /*#__PURE__*/React.createElement(Piece, {
    labels: t.labels,
    captions: t.captions
  })), /*#__PURE__*/React.createElement(TweaksPanel, null, /*#__PURE__*/React.createElement(TweakSection, {
    label: "Editor"
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Motion editor",
    value: t.motionEditor,
    onChange: v => setTweak('motionEditor', v)
  }), /*#__PURE__*/React.createElement(TweakSection, {
    label: "Overlays"
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Scientific labels",
    value: t.labels,
    onChange: v => setTweak('labels', v)
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Captions",
    value: t.captions,
    onChange: v => setTweak('captions', v)
  })));
}
window.PmhcVideoV2 = PmhcVideo;
;
/* src/animations/submission-film/act0-coldopen.jsx */
// Submission film — act 0 (the headline, 6s) and act 2 (the task, 12s).
//
// Runs on animations-v3 in ONE shared scope alongside the pMHC intro scene and
// film-scene.jsx, so there is no import/export here. Assumed already defined by
// that scope: C (palette), FONT, MOTION {enter, draw, pop}, tw(T,a,b,ease),
// lerp, clamp, Easing, React.
// Both components are pure functions of local time `u = T - t0`; neither calls
// useComposition, so the shell can place them at any film-time offset. In the
// current running order ColdOpen is 0:00-0:06 and ActQuestion is 0:20-0:32.
// Stage is 1920x1080.
//
// Both acts take their wording from site/index.html, which is the film's spec:
// ColdOpen is the masthead (h1 + TL;DR), ActQuestion is the "The task" section.
// The shell draws the base wash and the vignette, so neither act draws its own
// vignette (two would stack into a visibly darker edge). Neither act loads an
// image any more, so there is no asset path to retarget here.

// ── Copy, lifted verbatim from site/index.html ────────────────────────────
// Masthead h1 (its two lines) and the TL;DR standfirst, hard-wrapped for the
// stage. Edit the site and this together; the film is the site's trailer.
const TITLE_KICKER = 'ANTIGEN PRESENTATION STABILITY';
const TITLE_QUESTION = 'Sequence Is All You Need?';
const TLDR = ['We tested language-model embeddings and predicted structures against', 'a supervised model trained on sequences.', 'None of them won.'];
const FW = 1920,
  FH = 1080; // stage

// ── Shared chrome, matching pmhc-scene-v2.jsx ─────────────────────────────
function FilmWash({
  op = 1
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: op,
      background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)'
    }
  });
}

// Small-caps eyebrow, the same device as the site's <p class="eyebrow">.
function FilmEyebrow({
  op,
  text,
  y,
  size = 22
}) {
  if (op <= 0.001) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: y,
      opacity: op,
      textAlign: 'center',
      font: `600 ${size}px ${FONT}`,
      color: '#5A5C6E',
      letterSpacing: '0.22em',
      textTransform: 'uppercase'
    }
  }, text);
}

// ══════════════════════════════════════════════════════════════════════════
// ACT 0 — THE HEADLINE (6s, film 0:00-0:06)
//
// The film's first frame, and it opens on the claim rather than on an image:
// the site masthead, already fully on screen at u = 0. Nothing fades up from
// black and nothing resolves out of a preceding shot, so the entrance is a
// settle, not an appearance — the kicker and the question are at full opacity
// in frame one and only breathe into place, then the crimson rule draws and
// the TL;DR arrives under it, with "None of them won." landing last.
//
// Everything that used to precede this (the groove and peptide stills, the
// compressed clock, the exponential decay and its sparkline, the ring pulse,
// and the hero-pair micro-type with its "decay illustrative" caption) was cut
// with the opening sequence. The honesty caption went with the imagery it
// qualified; there is no illustrated quantity left on screen to qualify.
// ══════════════════════════════════════════════════════════════════════════
const CO = {
  // Must match the shell's slot. The card is fully landed by ~1.7s (punch line
  // in at 1.70), so 6s left 4.3s of a motionless frame before the cut — long
  // enough to read as a stall on the film's opening.
  dur: 4.5,
  settle: 0.90,
  // the headline's breathe-into-place, from frame one
  rule: 0.30,
  // crimson rule draws
  tldr: 0.60,
  // the first two TL;DR lines
  punch: 1.05,
  // "None of them won." lands last
  out: 0.28,
  // dissolve into Act I's biology
  top: 336 // whole block, optically centred for a title card
};
function ColdOpen({
  T,
  t0
}) {
  // L unused: no labelled chrome left in this act
  const u = T - (t0 || 0);

  // A settle, not an entrance: these start at full opacity so frame zero is
  // already the headline. Only the scale, the drift and the tracking move.
  const set = tw(u, 0, CO.settle, Easing.easeOutCubic);
  const rule = tw(u, CO.rule, CO.rule + 0.75, MOTION.draw);
  const tldrP = tw(u, CO.tldr, CO.tldr + 0.65, Easing.linear);
  const punchP = tw(u, CO.punch, CO.punch + 0.65, MOTION.enter);
  const exit = 1 - tw(u, CO.dur - CO.out, CO.dur, Easing.linear);
  return /*#__PURE__*/React.createElement("div", {
    "data-screen-label": `headline t=${Math.floor(u)}s`,
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      opacity: exit,
      fontFamily: FONT
    }
  }, /*#__PURE__*/React.createElement(FilmWash, null), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: CO.top,
      textAlign: 'center',
      pointerEvents: 'none'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      font: `600 42px ${FONT}`,
      color: '#3A3C50',
      lineHeight: 1.1,
      letterSpacing: `${lerp(0.235, 0.17, set)}em`
    }
  }, TITLE_KICKER), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 26,
      transform: `translateY(${(1 - set) * 10}px) scale(${lerp(1.016, 1, set)})`,
      font: `600 104px ${FONT}`,
      color: C.ink,
      letterSpacing: '-0.005em',
      lineHeight: 1.04
    }
  }, TITLE_QUESTION), /*#__PURE__*/React.createElement("div", {
    style: {
      width: lerp(0, 480, rule),
      height: 4,
      background: C.crimson,
      margin: '30px auto 0'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 30,
      font: `400 29px ${FONT}`,
      color: '#4A4C5E',
      lineHeight: 1.52
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      opacity: tldrP
    }
  }, TLDR[0]), /*#__PURE__*/React.createElement("div", {
    style: {
      opacity: tldrP
    }
  }, TLDR[1]), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 14,
      opacity: punchP,
      transform: `translateY(${(1 - punchP) * 8}px)`,
      font: `600 29px ${FONT}`,
      color: C.ink
    }
  }, TLDR[2]))));
}

// ══════════════════════════════════════════════════════════════════════════
// ACT 2 — THE TASK (12s, film 0:22-0:34)
//
// site/index.html's "The task" section, set as type. Deliberately the calmest
// frame in the film: one headline, two relations, two explanatory lines, on
// near-empty wash. No plot, no watermark, no second layer.
//
// The point the colour makes: k_off appears in BOTH relations (crimson in
// both), but affinity also divides by k_on (teal, affinity only). They are
// different quantities, so a model fitted to one is not fitted to the other.
// ══════════════════════════════════════════════════════════════════════════

// Formulas are laid out on an explicit monospace grid rather than left to the
// text engine, so the k_off / k_on underlines below can be placed from the same
// arithmetic and land exactly under their terms. `s` marks a subscript run and
// `g` tags the term the colour and the underline belong to.
const FORM_SIZE = 52,
  FORM_ADV = FORM_SIZE * 0.6,
  FORM_SUB = 0.62;
const FORM_SUB_DY = FORM_SIZE * 0.52; // subscript box top, so it sits on the baseline
const FORM_RULE_DY = FORM_SIZE * 1.26; // underline, clear of the subscript descender

const FORM_HALF = [{
  t: 't'
}, {
  t: '½',
  s: 1
}, {
  t: ' = ln 2 / '
}, {
  t: 'k',
  g: 'koff'
}, {
  t: 'off',
  s: 1,
  g: 'koff'
}];
const FORM_AFF = [{
  t: 'K'
}, {
  t: 'd',
  s: 1
}, {
  t: ' = '
}, {
  t: 'k',
  g: 'koff'
}, {
  t: 'off',
  s: 1,
  g: 'koff'
}, {
  t: ' / '
}, {
  t: 'k',
  g: 'kon'
}, {
  t: 'on',
  s: 1,
  g: 'kon'
}];
function formLayout(tokens) {
  let x = 0;
  const parts = tokens.map(tk => {
    const w = tk.t.length * FORM_ADV * (tk.s ? FORM_SUB : 1);
    const part = {
      ...tk,
      x,
      w
    };
    x += w;
    return part;
  });
  const span = g => {
    const hit = parts.filter(p => p.g === g);
    if (!hit.length) return null;
    return {
      x0: Math.min(...hit.map(p => p.x)),
      x1: Math.max(...hit.map(p => p.x + p.w))
    };
  };
  return {
    parts,
    w: x,
    koff: span('koff'),
    kon: span('kon')
  };
}
const TASK_GROUP_COLOR = {
  koff: C.crimson,
  kon: C.tcrA
};

// Two-column block, centred as a whole: labels right-aligned into the gutter,
// formulas left-aligned out of it. Both relations open with one glyph plus one
// subscript, so a shared left edge also aligns the two equals signs.
const TASK_LABEL_RIGHT = 1012,
  TASK_FORM_X = 1092;
const TASK_ROWS = [{
  y: 498,
  label: 'Stability — how long it stays bound',
  tokens: FORM_HALF
}, {
  y: 624,
  label: 'Affinity — how readily it binds',
  tokens: FORM_AFF
}];

// An inline k_off / k_on for the prose lines, coloured to match the formulas.
function Chem({
  sym,
  sub,
  color,
  size = 30
}) {
  return /*#__PURE__*/React.createElement("span", {
    style: {
      color,
      fontWeight: 600,
      whiteSpace: 'nowrap'
    }
  }, sym, /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: Math.round(size * 0.62),
      verticalAlign: 'sub'
    }
  }, sub));
}
function TaskRow({
  row,
  p,
  koffP,
  konP
}) {
  if (p <= 0.001) return null;
  const L = formLayout(row.tokens);
  const dy = (1 - p) * 14;
  const ruleFor = (span, prog, color) => span && prog > 0.004 ? /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: span.x0,
      top: FORM_RULE_DY,
      height: 3.5,
      width: (span.x1 - span.x0) * prog,
      background: color,
      borderRadius: 2
    }
  }) : null;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      top: row.y + 8,
      width: TASK_LABEL_RIGHT,
      textAlign: 'right',
      opacity: p,
      transform: `translateY(${dy}px)`,
      font: `400 30px ${FONT}`,
      color: '#4A4C5E',
      lineHeight: 1
    }
  }, row.label), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: TASK_FORM_X,
      top: row.y,
      width: L.w,
      height: FORM_SIZE * 1.6,
      opacity: p,
      transform: `translateY(${dy}px)`
    }
  }, L.parts.map((pt, i) => /*#__PURE__*/React.createElement("span", {
    key: i,
    style: {
      position: 'absolute',
      left: pt.x,
      top: pt.s ? FORM_SUB_DY : 0,
      font: `500 ${Math.round(FORM_SIZE * (pt.s ? FORM_SUB : 1))}px ${FONT}`,
      color: TASK_GROUP_COLOR[pt.g] || C.ink,
      lineHeight: 1,
      whiteSpace: 'pre'
    }
  }, pt.t)), ruleFor(L.koff, koffP, C.crimson), ruleFor(L.kon, konP, C.tcrA)));
}
const AQ = {
  eyebrow: 0.30,
  head: 0.60,
  rowA: 2.10,
  rowB: 3.90,
  // each row rises in, then its terms get underlined
  tie: 6.50,
  // both depend on k_off; affinity also divides by k_on
  close: 9.10 // therefore one model does not give you the other
};
function ActQuestion({
  T,
  t0,
  L
}) {
  const u = T - (t0 || 0);
  const eyeP = tw(u, AQ.eyebrow, AQ.eyebrow + 0.55, Easing.linear);
  const headP = tw(u, AQ.head, AQ.head + 0.95, MOTION.enter);
  const hairP = tw(u, AQ.head + 0.55, AQ.head + 1.25, MOTION.draw);
  const rowAP = tw(u, AQ.rowA, AQ.rowA + 0.80, MOTION.enter);
  const rowBP = tw(u, AQ.rowB, AQ.rowB + 0.80, MOTION.enter);
  // k_off is underlined in both rows — same colour, same quantity.
  const koffA = tw(u, AQ.rowA + 0.95, AQ.rowA + 1.45, MOTION.draw);
  const koffB = tw(u, AQ.rowB + 0.95, AQ.rowB + 1.45, MOTION.draw);
  const konB = tw(u, AQ.rowB + 1.55, AQ.rowB + 2.05, MOTION.draw);
  const tieP = tw(u, AQ.tie, AQ.tie + 0.70, Easing.linear);
  const closeP = tw(u, AQ.close, AQ.close + 0.70, Easing.linear);
  // Same hand-off as the cold open: the shell holds an act 0.3s past its slot,
  // so dissolve rather than sit opaque over the next act's opening frames.
  const exit = 1 - tw(u, 11.90, 12.18, Easing.linear);
  return /*#__PURE__*/React.createElement("div", {
    "data-screen-label": `task t=${Math.floor(u)}s`,
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      opacity: exit,
      fontFamily: FONT
    }
  }, /*#__PURE__*/React.createElement(FilmWash, null), /*#__PURE__*/React.createElement(FilmEyebrow, {
    op: eyeP,
    text: "the task",
    y: 252
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 306,
      textAlign: 'center',
      opacity: headP,
      transform: `translateY(${(1 - headP) * 16}px)`,
      font: `600 62px ${FONT}`,
      color: C.ink,
      letterSpacing: '-0.005em',
      lineHeight: 1.1
    }
  }, "Stability is different from affinity."), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: FW / 2 - 90 * hairP,
      top: 416,
      width: 180 * hairP,
      height: 2,
      background: C.ink,
      opacity: 0.28
    }
  }), TASK_ROWS.map((row, i) => /*#__PURE__*/React.createElement(TaskRow, {
    key: i,
    row: row,
    p: i === 0 ? rowAP : rowBP,
    koffP: i === 0 ? koffA : koffB,
    konP: i === 0 ? 0 : konB
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 790,
      textAlign: 'center',
      opacity: tieP,
      font: `400 30px ${FONT}`,
      color: C.ink,
      lineHeight: 1.5
    }
  }, "Both depend on ", /*#__PURE__*/React.createElement(Chem, {
    sym: "k",
    sub: "off",
    color: C.crimson
  }), ". Affinity also divides it by", ' ', /*#__PURE__*/React.createElement(Chem, {
    sym: "k",
    sub: "on",
    color: C.tcrA
  }), " \u2014 half-life does not."), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 862,
      textAlign: 'center',
      opacity: closeP,
      font: `400 30px ${FONT}`,
      color: '#4A4C5E',
      lineHeight: 1.5
    }
  }, "A model that predicts one does not automatically predict the other."));
}
window.ColdOpen = ColdOpen;
window.ActQuestion = ActQuestion;
;
/* src/animations/submission-film/act2-approach.jsx */
// Act 2 — OUR APPROACH. 12 s (film 0:34–0:46). The animated form of the
// "Our approach" section of site/index.html: the ten arms we tested, grouped
// into the three families of method they come from, under one protocol.
//
// This beat is SETUP, not result. It deliberately carries no score, no
// ranking, no winner and no outcome colour-coding — the verdict act that
// follows is the first time a number appears. Everything here is ink or grey;
// crimson (#B22222) stays reserved for the +0.05 threshold rule the verdict
// act brings on, so the first crimson mark in the film is that line.
//
// Artifact-runtime file: no import / no export, one shared scope evaluated by
// index.html in a single indirect eval. React, Easing, clamp and the intro
// scene's C / FONT / MOTION / tw / lerp are assumed already defined. Every
// module-scope symbol here is AP_* so nothing collides with pmhc-scene-v2.jsx
// or a sibling act in that same scope.
//
// Renders from `T` only — no effects, no rAF, no Math.random — so any seeked
// frame is a deterministic render, which is the exporter's contract.
//
// DATA. The ten arm names are the site's list, verbatim, and are a static
// editorial list: hardcoding them is correct. The only live number is how
// many arms reached the held-out test, read from window.__FILM__.arms.length
// (6) with a guard, so this act and the verdict act can never disagree about
// how many rows the result table has.

const AP_DUR = 12;
const AP_INK = typeof C !== 'undefined' && C.ink || '#2B2D42';
const AP_GREY = '#6B6D7E';
const AP_FAINT_TX = '#8A8C9C';
const AP_BG = '#EDEEF2';
const AP_FONT = typeof FONT !== 'undefined' && FONT || '"IBM Plex Mono", ui-monospace, monospace';
const AP_LIN = Easing.linear,
  AP_IO = Easing.easeInOutCubic,
  AP_OUT = Easing.easeOutCubic;

// ── layout ────────────────────────────────────────────────────────────────
// IBM Plex Mono advances at 0.6 em, so every width below is countable and the
// column fits were checked by hand rather than measured at runtime.
const AP_L = 140,
  AP_R = 1780;
const AP_COLW = 506;
const AP_COLX = [140, 707, 1274]; // three families, three columns
const AP_HEADY = 312; // family label baseline
const AP_HRULE = 326; // rule under the family label
const AP_DESCY = 356; // one-line family descriptor
const AP_ROW0 = 404,
  AP_PITCH = 78; // arm rows inside a column
const AP_NOTE_DY = 29; // qualifier baseline, below the name
const AP_NAME_DX = 52; // name column, right of the index
const AP_COLTOP = 288,
  AP_COLBOT = 694; // the vertical dividers' extent
const AP_BRK = 744; // the "one protocol" bracket
const AP_LEADY = 818,
  AP_SUBY = 860,
  AP_FOOTY = 938;
const AP_MID = (AP_L + AP_R) / 2;

// ── beats (local seconds) ─────────────────────────────────────────────────
// Entrance is staggered BY FAMILY, 2.2 s apart, so the viewer reads three
// blocks arriving rather than ten rows scrolling. The last row lands at 7.1 s
// and the closing line is complete by 10.1 s, leaving ~2 s on the full list —
// the frame this beat exists to hand to the verdict act.
// The preceding act has faded itself out by ~0.15 s into this one, so the
// header has to be up almost immediately or the cut shows a blank grey hole.
const APT = {
  tag: 0.02,
  title: 0.05,
  rule: 0.30,
  fam: [0.80, 3.00, 5.20],
  famIn: 0.45,
  armLead: 0.55,
  armStep: 0.28,
  armIn: 0.50,
  brk: 7.60,
  lead: 8.30,
  sub: 8.75,
  foot: 9.55
};

// ── the ten arms, exactly as site/index.html lists them ───────────────────
// Split into {n: name, q: qualifier} only for typography; concatenating n and
// q reproduces the site's line. The descriptors say what the family IS, not
// how it did.
const AP_FAMS = [{
  id: 'I',
  label: 'SEQUENCE',
  mark: 'square',
  desc: 'learned from the labels alone',
  arms: [{
    n: 'Allele-mean',
    q: 'sanity floor'
  }, {
    n: 'Single sequence MLP',
    q: 'peptide + HLA contact residues'
  }, {
    n: 'Sequence ensemble',
    q: '30 networks'
  }]
}, {
  id: 'II',
  label: 'PRETRAINED EMBEDDINGS',
  mark: 'circle',
  desc: 'features from a protein language model',
  arms: [{
    n: 'ESM-2 35M embeddings',
    q: 'alone'
  }, {
    n: 'ESM-2 150M embeddings',
    q: 'alone'
  }, {
    n: 'ESM-2 + sequence features',
    q: ''
  }]
}, {
  id: 'III',
  label: 'STRUCTURE & ENERGY',
  mark: 'diamond',
  desc: 'features from a predicted 3-D complex',
  arms: [{
    n: 'Boltz-2 structure',
    q: 'geometry only'
  }, {
    n: 'Boltz-2 structure',
    q: 'confidence only'
  }, {
    n: 'Boltz-2 structure',
    q: 'seq + geometry + confidence'
  }, {
    n: 'FoldX energy + sequence',
    q: ''
  }]
}];
const AP_TOTAL = AP_FAMS.reduce((s, f) => s + f.arms.length, 0); // 10

const AP_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
const apWord = n => n >= 0 && n < AP_WORDS.length ? AP_WORDS[n] : String(n);
const apCap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
const ap2 = n => n < 10 ? '0' + n : String(n);

// How many arms were carried through to the held-out test. The verdict act
// draws one row per entry in the same array, so this can never drift from the
// table the next beat shows. Falls back to the known 6 if the data block is
// missing (the page runs before film_data.js has been generated).
function apTested() {
  const film = typeof window !== 'undefined' && window.__FILM__ || null;
  const n = film && Array.isArray(film.arms) ? film.arms.length : 0;
  return n > 0 ? n : 6;
}

// ── svg text ──────────────────────────────────────────────────────────────
function AP_Tx({
  x,
  y,
  op,
  size,
  weight,
  color,
  anchor,
  track,
  children
}) {
  if (op != null && op <= 0.004) return null;
  return /*#__PURE__*/React.createElement("text", {
    x: x,
    y: y,
    opacity: op == null ? 1 : op,
    textAnchor: anchor || 'start',
    style: {
      fontFamily: AP_FONT,
      fontSize: (size || 22) + 'px',
      fontWeight: weight || 500,
      fill: color || AP_INK,
      letterSpacing: track || '0',
      fontVariantNumeric: 'tabular-nums',
      whiteSpace: 'pre'
    }
  }, children);
}

// Three family glyphs, so the families stay distinguishable without colour.
// Filled / hollow / outlined-rotated, ink only.
function AP_Mark({
  kind,
  cx,
  cy,
  op,
  s = 7
}) {
  if (op <= 0.004) return null;
  if (kind === 'square') {
    return /*#__PURE__*/React.createElement("rect", {
      x: cx - s,
      y: cy - s,
      width: s * 2,
      height: s * 2,
      fill: AP_INK,
      opacity: op
    });
  }
  if (kind === 'circle') {
    return /*#__PURE__*/React.createElement("circle", {
      cx: cx,
      cy: cy,
      r: s,
      fill: "none",
      stroke: AP_INK,
      strokeWidth: "2.6",
      opacity: op
    });
  }
  const d = s * 1.28;
  return /*#__PURE__*/React.createElement("polygon", {
    points: `${cx},${cy - d} ${cx + d},${cy} ${cx},${cy + d} ${cx - d},${cy}`,
    fill: "none",
    stroke: AP_INK,
    strokeWidth: "2.6",
    opacity: op
  });
}

// ── one arm row ───────────────────────────────────────────────────────────
// Enters as fade + a 14 px rise, with a short rule drawing out from the index
// gutter — the row "lands" on the list rather than popping in.
function AP_Row({
  t,
  at,
  x,
  idx,
  name,
  note
}) {
  const p = tw(t, at, at + APT.armIn, AP_OUT);
  if (p <= 0.004) return null;
  const dy = lerp(14, 0, p);
  const tick = tw(t, at, at + APT.armIn * 0.8, AP_IO);
  // The row's own origin is y = 0; the caller has already translated it onto
  // its line, so these offsets are read relative to the name's baseline.
  return /*#__PURE__*/React.createElement("g", {
    transform: `translate(0,${dy.toFixed(2)})`,
    opacity: p
  }, /*#__PURE__*/React.createElement("line", {
    x1: x,
    y1: "7",
    x2: x + 26 * tick,
    y2: "7",
    stroke: AP_INK,
    strokeWidth: "1.6",
    opacity: "0.35"
  }), /*#__PURE__*/React.createElement(AP_Tx, {
    x: x,
    y: -2,
    size: 17,
    weight: 500,
    color: AP_FAINT_TX,
    track: "0.06em"
  }, idx), /*#__PURE__*/React.createElement(AP_Tx, {
    x: x + AP_NAME_DX,
    y: 0,
    size: 24,
    weight: 500
  }, name), note ? /*#__PURE__*/React.createElement(AP_Tx, {
    x: x + AP_NAME_DX,
    y: AP_NOTE_DY,
    size: 20,
    weight: 400,
    color: AP_GREY
  }, note) : null);
}

// ── one family column ─────────────────────────────────────────────────────
function AP_Family({
  t,
  fam,
  x,
  at,
  first
}) {
  const h = tw(t, at, at + APT.famIn, AP_OUT);
  const rule = tw(t, at + 0.12, at + 0.72, AP_IO);
  const div = tw(t, at - 0.15, at + 0.5, AP_IO);
  if (h <= 0.004 && div <= 0.004) return null;
  const labelX = x + 30;
  return /*#__PURE__*/React.createElement("g", null, !first && div > 0.004 && /*#__PURE__*/React.createElement("line", {
    x1: x - 31,
    y1: AP_COLTOP,
    x2: x - 31,
    y2: lerp(AP_COLTOP, AP_COLBOT, div),
    stroke: AP_INK,
    strokeWidth: "1",
    opacity: "0.16"
  }), h > 0.004 && /*#__PURE__*/React.createElement("g", {
    opacity: h,
    transform: `translate(0,${lerp(10, 0, h).toFixed(2)})`
  }, /*#__PURE__*/React.createElement(AP_Mark, {
    kind: fam.mark,
    cx: x + 8,
    cy: AP_HEADY - 7,
    op: h
  }), /*#__PURE__*/React.createElement(AP_Tx, {
    x: labelX,
    y: AP_HEADY,
    size: 19,
    weight: 600,
    track: "0.16em"
  }, fam.label), /*#__PURE__*/React.createElement(AP_Tx, {
    x: x + AP_COLW,
    y: AP_HEADY,
    size: 17,
    weight: 500,
    color: AP_FAINT_TX,
    anchor: "end",
    track: "0.14em"
  }, fam.arms.length + ' ARMS'), /*#__PURE__*/React.createElement(AP_Tx, {
    x: labelX,
    y: AP_DESCY,
    size: 18,
    weight: 400,
    color: AP_GREY
  }, fam.desc)), rule > 0.004 && /*#__PURE__*/React.createElement("line", {
    x1: x,
    y1: AP_HRULE,
    x2: lerp(x, x + AP_COLW, rule),
    y2: AP_HRULE,
    stroke: AP_INK,
    strokeWidth: "1.5",
    opacity: 0.45 * rule
  }));
}

// ── the bracket that ties the three columns to one protocol ───────────────
function AP_Bracket({
  p
}) {
  if (p <= 0.004) return null;
  const halfSpan = (AP_R - AP_L) / 2;
  const h = halfSpan * p;
  const drop = clamp((p - 0.6) / 0.4, 0, 1);
  return /*#__PURE__*/React.createElement("g", {
    opacity: Math.min(1, p * 2)
  }, /*#__PURE__*/React.createElement("line", {
    x1: AP_MID - h,
    y1: AP_BRK,
    x2: AP_MID + h,
    y2: AP_BRK,
    stroke: AP_INK,
    strokeWidth: "1.8"
  }), /*#__PURE__*/React.createElement("line", {
    x1: AP_MID - h,
    y1: AP_BRK - 14,
    x2: AP_MID - h,
    y2: AP_BRK,
    stroke: AP_INK,
    strokeWidth: "1.8"
  }), /*#__PURE__*/React.createElement("line", {
    x1: AP_MID + h,
    y1: AP_BRK - 14,
    x2: AP_MID + h,
    y2: AP_BRK,
    stroke: AP_INK,
    strokeWidth: "1.8"
  }), drop > 0.004 && /*#__PURE__*/React.createElement("line", {
    x1: AP_MID,
    y1: AP_BRK,
    x2: AP_MID,
    y2: AP_BRK + 20 * drop,
    stroke: AP_INK,
    strokeWidth: "1.8"
  }));
}

// ── the act ───────────────────────────────────────────────────────────────
function ActApproach({
  T,
  t0,
  L
}) {
  const raw = T - (t0 || 0);
  if (raw < -0.3) return null;
  const t = Math.min(raw, AP_DUR); // hold the final frame
  const labels = L !== false;
  const fade = tw(t, 0, 0.14, AP_LIN);
  const tagP = tw(t, APT.tag, APT.tag + 0.36, AP_LIN);
  const titleP = tw(t, APT.title, APT.title + 0.57, AP_OUT);
  const ruleP = tw(t, APT.rule, APT.rule + 0.65, AP_IO);
  const brkP = tw(t, APT.brk, APT.brk + 0.85, AP_IO);
  const leadP = tw(t, APT.lead, APT.lead + 0.5, AP_OUT);
  const subP = tw(t, APT.sub, APT.sub + 0.5, AP_OUT);
  const footP = tw(t, APT.foot, APT.foot + 0.55, AP_LIN);

  // when each arm row arrives, and therefore what the counter reads
  const armAt = [];
  AP_FAMS.forEach((f, fi) => {
    f.arms.forEach((a, ai) => {
      armAt.push(APT.fam[fi] + APT.armLead + ai * APT.armStep);
    });
  });
  let landed = 0;
  armAt.forEach(a => {
    if (t >= a + APT.armIn * 0.55) landed++;
  });
  const tested = apTested();
  return /*#__PURE__*/React.createElement("div", {
    "data-screen-label": `approach t=${Math.floor(t)}s`,
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      background: AP_BG,
      fontFamily: AP_FONT,
      opacity: fade
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "1920",
    height: "1080",
    style: {
      position: 'absolute',
      inset: 0
    }
  }, tagP > 0.004 && /*#__PURE__*/React.createElement("g", {
    opacity: tagP
  }, /*#__PURE__*/React.createElement(AP_Tx, {
    x: AP_L,
    y: 103,
    size: 22,
    weight: 600,
    track: "0.06em"
  }, "OUR APPROACH"), /*#__PURE__*/React.createElement("line", {
    x1: AP_L + 200,
    y1: 96,
    x2: AP_L + 236,
    y2: 96,
    stroke: AP_INK,
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement(AP_Tx, {
    x: AP_L + 256,
    y: 103,
    size: 22,
    weight: 500,
    track: "0.06em"
  }, "WHAT WE TESTED")), /*#__PURE__*/React.createElement("g", {
    opacity: titleP,
    transform: `translate(0,${lerp(12, 0, titleP).toFixed(2)})`
  }, /*#__PURE__*/React.createElement(AP_Tx, {
    x: AP_L,
    y: 206,
    size: 46,
    weight: 600
  }, "Ten arms. Three families of method.")), titleP > 0.004 && /*#__PURE__*/React.createElement("g", {
    opacity: titleP
  }, /*#__PURE__*/React.createElement(AP_Tx, {
    x: AP_R,
    y: 172,
    size: 17,
    weight: 500,
    color: AP_FAINT_TX,
    anchor: "end",
    track: "0.18em"
  }, "ARMS TESTED"), /*#__PURE__*/React.createElement(AP_Tx, {
    x: AP_R,
    y: 220,
    size: 42,
    weight: 600,
    anchor: "end"
  }, ap2(landed) + ' / ' + AP_TOTAL)), /*#__PURE__*/React.createElement("line", {
    x1: AP_L,
    y1: 246,
    x2: lerp(AP_L, AP_R, ruleP),
    y2: 246,
    stroke: AP_INK,
    strokeWidth: "1.5",
    opacity: 0.3 * ruleP
  }), AP_FAMS.map((f, fi) => /*#__PURE__*/React.createElement(AP_Family, {
    key: f.label,
    t: t,
    fam: f,
    x: AP_COLX[fi],
    at: APT.fam[fi],
    first: fi === 0
  })), (() => {
    const out = [];
    let g = 0;
    AP_FAMS.forEach((f, fi) => {
      f.arms.forEach((a, ai) => {
        g++;
        out.push(/*#__PURE__*/React.createElement("g", {
          key: f.label + ai,
          transform: `translate(0,${AP_ROW0 + ai * AP_PITCH})`
        }, /*#__PURE__*/React.createElement(AP_Row, {
          t: t,
          at: APT.fam[fi] + APT.armLead + ai * APT.armStep,
          x: AP_COLX[fi],
          idx: ap2(g),
          name: a.n,
          note: a.q
        })));
      });
    });
    return out;
  })(), /*#__PURE__*/React.createElement(AP_Bracket, {
    p: brkP
  }), labels && /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("g", {
    opacity: leadP,
    transform: `translate(0,${lerp(10, 0, leadP).toFixed(2)})`
  }, /*#__PURE__*/React.createElement(AP_Tx, {
    x: AP_MID,
    y: AP_LEADY,
    size: 34,
    weight: 600,
    anchor: "middle"
  }, "One protocol for all ten.")), /*#__PURE__*/React.createElement(AP_Tx, {
    x: AP_MID,
    y: AP_SUBY,
    op: subP,
    size: 26,
    weight: 400,
    color: AP_GREY,
    anchor: "middle"
  }, 'the same rows  ·  the same 30-network ensembling  ·  an equal tuning budget'), /*#__PURE__*/React.createElement(AP_Tx, {
    x: AP_MID,
    y: AP_FOOTY,
    op: footP * 0.95,
    size: 21,
    weight: 400,
    color: AP_FAINT_TX,
    anchor: "middle"
  }, apCap(apWord(tested)) + ' of the ten were carried to the held-out test —' + ' FoldX and ESM-2 150M were not.'))));
}
window.ActApproach = ActApproach;
;
/* src/animations/submission-film/act5-verdict.jsx */
// Act 5 — THE VERDICT. 20 s. The film's conclusion: the +0.05 bar is declared
// first, six arms' test intervals are measured against it, and the act ends
// held on the one conclusive negative, alone and entirely below zero.
//
// Artifact-runtime file: no import / no export, one shared scope. It assumes
// React, Easing, clamp, and the pmhc-intro scene's C / FONT / MOTION / tw /
// lerp are already evaluated. Every identifier here is AV_*/av* so nothing
// collides with pmhc-scene-v2.jsx or the sibling acts in that same scope.
//
// Style: instrument, not photograph. Ink on a flat grey wash, hairlines, IBM
// Plex Mono, no gradients, no shadows, no rounded corners. Geometry is one
// <svg>; only the caption is a div (to match act 4's caption block).
//
// Everything renders from T only (no effects, no rAF, no Math.random) so a
// seeked frame is a deterministic render — the exporter's contract.
//
// DATA. Every number comes from window.__FILM__.arms at render time; nothing
// about the result is written into this file. Each arm supplies
//   name, is_baseline, spearman,
//   paired: {delta, ci_low, ci_high, verdict}   — NULL on the baseline arm.
// The scoring context also reads n_rows / n_alleles off an arm and the
// resample count off window.__FILM__.bootstrap.n_boot, which is emitted only
// when every headline contrast agrees — so it is printed only if present.
// The baseline arm has no paired block, so it gets no whisker and no delta
// readout; it IS the zero line. If window.__FILM__.arms is missing the act
// renders nothing rather than inventing a figure. Rounding happens only at
// the point of rendering.
//
// `L` gates the annotation layer (captions, verdict wording, the held-arm
// readout, the cost legend). Names, numbers, axes and tick labels always
// render — without them the chart is not a chart.

const AV_DUR = 20;
const AV_INK = typeof C !== 'undefined' && C.ink || '#2B2D42';
const AV_CRIM = typeof C !== 'undefined' && C.crimson || '#B22222';
const AV_GREY = '#6B6D7E';
const AV_GOOD = '#2E6E43';
const AV_BG = '#EDEEF2';
const AV_HAIR = 'rgba(43,45,66,0.20)';
const AV_FAINT = 'rgba(43,45,66,0.075)';
const AV_PANEL = '#F7F8FB';
const AV_FONT = typeof FONT !== 'undefined' && FONT || '"IBM Plex Mono", ui-monospace, monospace';
const AV_LIN = Easing.linear,
  AV_IO = Easing.easeInOutCubic,
  AV_OUT = Easing.easeOutCubic;

// ── layout ────────────────────────────────────────────────────────────────
const AV_L = 160,
  AV_R = 1760; // outer margins, as in act 4
const AV_NAMEX = 640,
  AV_NUMX = 790; // right-aligned text columns
const AV_X0 = 830,
  AV_X1 = 1740; // bar / whisker plot
const AV_ROW0 = 320,
  AV_PITCH = 92,
  AV_AXY = 820;
const AV_RAIL = 200; // where the threshold docks
// The top band is three reserved slots. Every label annotating the delta view
// lands in exactly one of them, and the view header yields to the gap bracket,
// so no two of them are ever on the same line.
const AV_BAND = 172; // view header (left) / gap headline (right)
const AV_SUB = 202; // second line of the gap annotation
const AV_BRK = 240; // the gap bracket rule, clear of both
// The held-arm card sits in the one rectangle of the delta view that no
// interval reaches: left of every whisker, right of the name columns.
// It expands once the control finding arrives; by then the other rows have
// cleared, so the wide state occludes nothing that is still being read.
const AV_CARD = {
  x: AV_X0 + 4,
  y: 300,
  w: 416,
  h: 252,
  w2: 756,
  h2: 276
};
// Where the rule is declared, before it docks. This act now states the bar
// itself — the firewall act that used to declare it is out of the cut.
const AV_DX0 = 300,
  AV_DX1 = 1650,
  AV_DY = 560;

// ── beats (local seconds) ─────────────────────────────────────────────────
const AVT = {
  rule: 0.30,
  // the bar is stated, before any score exists
  stmt: 1.15,
  why: 2.15,
  ctx: 3.35,
  dock: 5.15,
  // it docks to the top rail; the frame builds
  bars: 5.95,
  step: 0.32,
  grow: 0.80,
  morph: 9.85,
  morphE: 10.85,
  // bars collapse into differences
  vrd: 11.25,
  vstep: 0.52,
  // one verdict at a time, top to bottom
  gap: 14.60,
  dim: 15.90,
  hold: 16.30,
  ctrl: 17.60
  // there is no release: the act ends held on the conclusion, and the last
  // frame is what the film cuts away from.
};

// ── formatting (display only; the data path stays unrounded) ──────────────
const AV_MINUS = '−';
const avFix = (v, d) => (v < 0 ? AV_MINUS : '') + Math.abs(v).toFixed(d);
const avSig = (v, d) => (v < 0 ? AV_MINUS : '+') + Math.abs(v).toFixed(d);
const avGrp = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
// paired.verdict is the data's own wording; take its leading clause for a chip
// and normalise the ASCII double hyphen for the full-sentence readout.
const avClause = s => {
  if (!s) return '';
  const parts = String(s).split(/[,:]|\s--\s/).map(x => x.trim()).filter(Boolean);
  // "rules out a 0.05 gain" is the clause that decides; fall back to the first
  for (let i = 0; i < parts.length; i++) {
    if (parts[i].indexOf('rules out') >= 0) return parts[i].replace(/^and\s+/, '');
  }
  return parts.length ? parts[0] : '';
};
const avSentence = s => s ? String(s).replace(/\s--\s/g, ' — ') : '';
// a round-ish tick step for a span, so axes stay legible whatever the numbers
const avStep = (span, target) => {
  const r = Math.abs(span) / Math.max(1, target);
  if (!isFinite(r) || r <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log(r) / Math.LN10));
  const u = r / mag;
  return (u <= 1 ? 1 : u <= 2 ? 2 : u <= 5 ? 5 : 10) * mag;
};

// ── the data contract ─────────────────────────────────────────────────────
function avData() {
  const film = typeof window !== 'undefined' && window.__FILM__ || null;
  const raw = film && Array.isArray(film.arms) ? film.arms : null;
  if (!raw || !raw.length) return null;
  const num = v => {
    if (typeof v === 'string') {
      const p = parseFloat(v);
      return isFinite(p) ? p : NaN;
    }
    return typeof v === 'number' && isFinite(v) ? v : NaN;
  };
  const arms = [];
  for (let i = 0; i < raw.length; i++) {
    const a = raw[i] || {};
    const score = num(a.spearman);
    if (!isFinite(score)) continue;
    const p = a.paired && typeof a.paired === 'object' ? a.paired : null;
    let lo = p ? num(p.ci_low) : NaN,
      hi = p ? num(p.ci_high) : NaN;
    if (isFinite(lo) && isFinite(hi) && lo > hi) {
      const s = lo;
      lo = hi;
      hi = s;
    }
    const d = p ? num(p.delta) : NaN;
    arms.push({
      key: String(a.key || 'arm' + i),
      name: String(a.name || a.key || 'arm ' + (i + 1)),
      base: a.is_baseline === true || !p,
      score: score,
      delta: isFinite(d) ? d : 0,
      lo: lo,
      hi: hi,
      hasCI: isFinite(lo) && isFinite(hi),
      verdict: p && typeof p.verdict === 'string' ? p.verdict : '',
      nRows: num(a.n_rows),
      nAll: num(a.n_alleles)
    });
  }
  if (!arms.length) return null;
  arms.sort((x, y) => y.score - x.score); // the payload is sorted; be sure
  const thv = num(film.threshold);
  const th = isFinite(thv) && thv > 0 ? thv : 0.05;
  // the resample count is printed only if the payload carries it
  const boot = num(film.n_boot !== undefined ? film.n_boot : film.bootstrap && film.bootstrap.n_boot !== undefined ? film.bootstrap.n_boot : undefined);
  arms.forEach(a => {
    a.cls = a.base || !a.hasCI ? 'base' : a.hi < 0 ? 'worse' : a.hi < th ? 'short' : 'gain';
  });
  return {
    arms: arms,
    th: th,
    boot: boot
  };
}
const avCol = cls => cls === 'worse' ? AV_CRIM : cls === 'base' ? AV_INK : cls === 'gain' ? AV_GOOD : AV_GREY;

// ── svg text ──────────────────────────────────────────────────────────────
function AV_Tx({
  x,
  y,
  op,
  size,
  weight,
  color,
  anchor,
  track,
  halo,
  children
}) {
  if (op != null && op <= 0.004) return null;
  const st = {
    fontFamily: AV_FONT,
    fontSize: (size || 22) + 'px',
    fontWeight: weight || 500,
    fill: color || AV_INK,
    letterSpacing: track || '0',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'pre'
  };
  // knock the paper out around the glyphs: rules drawn under a number must not
  // cut through it
  if (halo) {
    st.stroke = AV_BG;
    st.strokeWidth = halo;
    st.paintOrder = 'stroke';
    st.strokeLinejoin = 'round';
  }
  return /*#__PURE__*/React.createElement("text", {
    x: x,
    y: y,
    opacity: op == null ? 1 : op,
    textAnchor: anchor || 'start',
    style: st
  }, children);
}

// ── the declared threshold: horizontal → docked → vertical at +0.05 ───────
// One object in three states. It arrives already drawn (act 3 left it up), so
// nothing about it animates on at t = 0.
function avThreshGeom(t, th, xb, rowsMid, span) {
  const p1 = tw(t, AVT.dock, AVT.dock + 0.70, AV_IO);
  const p2 = tw(t, AVT.morph, AVT.morphE, AV_IO);
  const A = {
    cx: (AV_DX0 + AV_DX1) / 2,
    cy: AV_DY,
    ang: 0,
    len: AV_DX1 - AV_DX0
  };
  const B = {
    cx: (AV_L + AV_R) / 2,
    cy: AV_RAIL,
    ang: 0,
    len: AV_R - AV_L
  };
  const Z = {
    cx: xb(th),
    cy: rowsMid,
    ang: 90,
    len: span
  };
  const mix = k => lerp(lerp(A[k], B[k], p1), Z[k], p2);
  return {
    cx: mix('cx'),
    cy: mix('cy'),
    ang: mix('ang'),
    len: mix('len'),
    p1: p1,
    p2: p2
  };
}
function AV_Threshold({
  t,
  th,
  g,
  op
}) {
  if (op <= 0.004) return null;
  const h = g.len / 2;
  const dr = tw(t, AVT.rule, AVT.rule + 1.05, AV_IO); // stated left to right
  if (dr <= 0.004) return null;
  const label = avSig(th, 2) + '  minimum worthwhile gain';
  const lw = label.length * 13.8; // 23px mono, so width is countable
  // the label never rotates: it walks between three anchors
  const la = [{
    x: AV_DX0,
    y: AV_DY - 26,
    a: 0
  }, {
    x: AV_R - lw,
    y: AV_BAND,
    a: 0
  }, {
    x: g.cx + 30,
    y: g.cy + h - 62,
    a: -90
  }];
  const lx = lerp(lerp(la[0].x, la[1].x, g.p1), la[2].x, g.p2);
  const ly = lerp(lerp(la[0].y, la[1].y, g.p1), la[2].y, g.p2);
  const lang = lerp(lerp(la[0].a, la[1].a, g.p1), la[2].a, g.p2);
  return /*#__PURE__*/React.createElement("g", {
    opacity: op
  }, /*#__PURE__*/React.createElement("g", {
    transform: `translate(${g.cx.toFixed(2)},${g.cy.toFixed(2)}) rotate(${g.ang.toFixed(3)})`
  }, /*#__PURE__*/React.createElement("line", {
    x1: -h,
    y1: "0",
    x2: -h + 2 * h * dr,
    y2: "0",
    stroke: AV_CRIM,
    strokeWidth: "3.5",
    strokeDasharray: "18 13"
  }), /*#__PURE__*/React.createElement("line", {
    x1: -h,
    y1: "-10",
    x2: -h,
    y2: "10",
    stroke: AV_CRIM,
    strokeWidth: "3"
  }), /*#__PURE__*/React.createElement("line", {
    x1: -h + 2 * h * dr,
    y1: "-10",
    x2: -h + 2 * h * dr,
    y2: "10",
    stroke: AV_CRIM,
    strokeWidth: "3"
  })), /*#__PURE__*/React.createElement("g", {
    transform: `translate(${lx.toFixed(2)},${ly.toFixed(2)}) rotate(${lang.toFixed(3)})`
  }, /*#__PURE__*/React.createElement(AV_Tx, {
    x: "0",
    y: "0",
    op: g.p1,
    size: 23,
    weight: 600,
    color: AV_CRIM
  }, label)));
}

// ── beat 1: the bar, declared before any score exists ────────────────────
// Wording and framing follow the project site's main-result and methods
// sections; the row and allele counts come from the payload.
function AV_Declare({
  t,
  th,
  out,
  ctx
}) {
  const a = tw(t, AVT.rule + 0.15, AVT.rule + 0.8, AV_OUT) * out;
  const b = tw(t, AVT.stmt, AVT.stmt + 0.7, AV_OUT) * out;
  const c = tw(t, AVT.why, AVT.why + 0.7, AV_OUT) * out;
  const d = tw(t, AVT.ctx, AVT.ctx + 0.7, AV_OUT) * out;
  if (a <= 0.004) return null;
  return /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_DX0,
    y: AV_DY - 150,
    op: a,
    size: 19,
    weight: 600,
    color: AV_GREY,
    track: "0.16em"
  }, "DECISION RULE \xB7 FIXED BEFORE ANY MODELLING"), /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_DX0,
    y: AV_DY - 78,
    op: b,
    size: 40,
    weight: 600,
    color: AV_CRIM
  }, avSig(th, 2) + ' median per-allele Spearman'), /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_DX0,
    y: AV_DY - 38,
    op: b,
    size: 23,
    color: AV_GREY
  }, "minimum worthwhile gain over the baseline"), /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_DX0,
    y: AV_DY + 62,
    op: c,
    size: 26
  }, "Below it, this benchmark cannot separate a real"), /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_DX0,
    y: AV_DY + 98,
    op: c,
    size: 26
  }, "difference from noise."), /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_DX0,
    y: AV_DY + 156,
    op: d,
    size: 18,
    color: AV_GREY,
    track: "0.14em"
  }, ctx));
}

// ── the axis: scores, then differences, on one horizontal rule ───────────
function AV_Axis({
  m,
  amax,
  xa,
  dmin,
  dmax,
  xb,
  frameP
}) {
  const aOp = (1 - m) * frameP;
  const bOp = m;
  const ticksA = [],
    stepA = avStep(amax, 5);
  for (let v = 0; v <= amax + 1e-9; v += stepA) ticksA.push(v);
  const ticksB = [],
    stepB = avStep(dmax - dmin, 5);
  for (let v = Math.ceil(dmin / stepB) * stepB; v <= dmax + 1e-9; v += stepB) ticksB.push(v);
  return /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("line", {
    x1: AV_X0,
    y1: AV_AXY,
    x2: AV_X1,
    y2: AV_AXY,
    stroke: AV_INK,
    strokeWidth: "2",
    opacity: 0.8 * frameP
  }), aOp > 0.004 && ticksA.map((v, i) => /*#__PURE__*/React.createElement("g", {
    key: 'a' + i,
    opacity: aOp
  }, /*#__PURE__*/React.createElement("line", {
    x1: xa(v),
    y1: AV_AXY,
    x2: xa(v),
    y2: AV_AXY + 9,
    stroke: AV_INK,
    strokeWidth: "1.5",
    opacity: "0.7"
  }), /*#__PURE__*/React.createElement(AV_Tx, {
    x: xa(v),
    y: AV_AXY + 34,
    size: 19,
    color: AV_GREY,
    anchor: "middle"
  }, v.toFixed(1)))), bOp > 0.004 && ticksB.map((v, i) => /*#__PURE__*/React.createElement("g", {
    key: 'b' + i,
    opacity: bOp
  }, /*#__PURE__*/React.createElement("line", {
    x1: xb(v),
    y1: AV_AXY,
    x2: xb(v),
    y2: AV_AXY + 9,
    stroke: AV_INK,
    strokeWidth: "1.5",
    opacity: "0.7"
  }), /*#__PURE__*/React.createElement(AV_Tx, {
    x: xb(v),
    y: AV_AXY + 34,
    size: 19,
    color: Math.abs(v) < 1e-9 ? AV_INK : AV_GREY,
    anchor: "middle"
  }, Math.abs(v) < 1e-9 ? '0' : avSig(v, 2)))));
}

// ── the six rows: bars grow, then collapse into intervals about zero ──────
function AV_Rows({
  t,
  arms,
  th,
  m,
  xa,
  xb,
  rowY,
  out,
  dimOf,
  emphOf,
  textOf,
  pulseOf
}) {
  if (out <= 0.004) return null;
  const zero = clamp((m - 0.15) / 0.5, 0, 1);
  const x0 = xa(0);
  return /*#__PURE__*/React.createElement("g", {
    opacity: out
  }, zero > 0.004 && /*#__PURE__*/React.createElement("line", {
    x1: xb(0),
    y1: AV_ROW0 - 60,
    x2: xb(0),
    y2: AV_AXY,
    stroke: AV_INK,
    strokeWidth: "2.5",
    opacity: zero * 0.9
  }), arms.map((a, i) => {
    const t0 = AVT.bars + i * AVT.step;
    const bp = tw(t, t0, t0 + AVT.grow, AV_IO);
    const wp = tw(t, t0 + AVT.grow - 0.15, t0 + AVT.grow + 0.5, AV_IO);
    if (bp <= 0.004) return null;
    const y = rowY(i);
    const dim = dimOf(i),
      emph = emphOf(i);
    const col = avCol(a.cls);
    // phase A: the paired interval re-centred on this arm's own score
    const aLo = a.hasCI ? a.score + (a.lo - a.delta) : a.score;
    const aHi = a.hasCI ? a.score + (a.hi - a.delta) : a.score;
    const pt = lerp(lerp(x0, xa(a.score), bp), xb(a.delta), m);
    const barL = lerp(x0, xb(a.delta), m);
    const barOp = (1 - clamp((m - 0.2) / 0.45, 0, 1)) * bp;
    const w0 = lerp(lerp(xa(a.score), xa(aLo), wp), xb(a.lo), m);
    const w1 = lerp(lerp(xa(a.score), xa(aHi), wp), xb(a.hi), m);
    const pulse = pulseOf(i);
    const cap = lerp(12, 17, m) + emph * 4 + pulse * 3;
    const lw = lerp(2.5, 3.5, m) + emph * 1.5 + pulse * 1.6;
    const ci = a.hasCI ? wp * clamp((m - 0.55) / 0.35, 0, 1) * textOf(i) : 0;
    return /*#__PURE__*/React.createElement("g", {
      key: a.key,
      opacity: dim
    }, barOp > 0.004 && /*#__PURE__*/React.createElement("rect", {
      x: barL,
      y: y - 17,
      width: Math.max(0, pt - barL),
      height: "34",
      fill: col,
      fillOpacity: a.base ? 0.5 : 0.34,
      stroke: col,
      strokeWidth: "2",
      opacity: barOp
    }), pulse > 0.01 && /*#__PURE__*/React.createElement("line", {
      x1: AV_NUMX + 24,
      y1: y,
      x2: Math.min(w0, pt) - 12,
      y2: y,
      stroke: col,
      strokeWidth: "1",
      opacity: pulse * 0.5
    }), a.hasCI && wp > 0.004 && /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("g", {
      stroke: col,
      strokeWidth: lw
    }, /*#__PURE__*/React.createElement("line", {
      x1: w0,
      y1: y,
      x2: w1,
      y2: y
    }), /*#__PURE__*/React.createElement("line", {
      x1: w0,
      y1: y - cap / 2,
      x2: w0,
      y2: y + cap / 2
    }), /*#__PURE__*/React.createElement("line", {
      x1: w1,
      y1: y - cap / 2,
      x2: w1,
      y2: y + cap / 2
    }))), /*#__PURE__*/React.createElement("circle", {
      cx: pt,
      cy: y,
      r: lerp(0, 6.5, m) + emph * 1.5 + pulse * 2,
      fill: col
    }), /*#__PURE__*/React.createElement(AV_Tx, {
      x: AV_NAMEX,
      y: y + 8,
      op: bp,
      size: 22,
      weight: a.base ? 600 : 500,
      anchor: "end"
    }, a.name), /*#__PURE__*/React.createElement(AV_Tx, {
      x: AV_NUMX,
      y: y + 9,
      op: bp * (1 - m),
      size: 27,
      weight: 600,
      anchor: "end"
    }, a.score.toFixed(3)), /*#__PURE__*/React.createElement(AV_Tx, {
      x: AV_NUMX,
      y: y + 9,
      op: m,
      size: 27,
      weight: 600,
      color: col,
      anchor: "end"
    }, a.hasCI ? avSig(a.delta, 3) : '0.000'), ci > 0.004 && /*#__PURE__*/React.createElement(AV_Tx, {
      x: (w0 + w1) / 2,
      y: y - 26,
      op: ci,
      size: 19,
      color: AV_GREY,
      anchor: "middle",
      halo: 6
    }, '[ ' + avFix(a.lo, 3) + '   ' + avFix(a.hi, 3) + ' ]'));
  }));
}

// the data's own verdict wording, one clause per row, under each interval
function AV_Verdicts({
  t,
  arms,
  xb,
  rowY,
  out,
  dimOf,
  textOf
}) {
  if (out <= 0.004) return null;
  return /*#__PURE__*/React.createElement("g", null, arms.map((a, i) => {
    const a0 = AVT.vrd + i * AVT.vstep;
    const p = tw(t, a0, a0 + 0.55, AV_OUT) * out * dimOf(i) * textOf(i);
    if (p <= 0.004) return null;
    // the top-scoring arm has no paired interval against itself; the site's
    // table calls that row what it is
    const txt = a.hasCI ? avClause(a.verdict) : i === 0 ? 'best in project' : 'baseline';
    if (!txt) return null;
    const cx = a.hasCI ? (xb(a.lo) + xb(a.hi)) / 2 : xb(a.delta);
    return /*#__PURE__*/React.createElement(AV_Tx, {
      key: a.key,
      x: cx,
      y: rowY(i) + 32,
      op: p,
      size: 18,
      color: a.hasCI ? avCol(a.cls) : AV_GOOD,
      weight: a.hasCI ? 500 : 600,
      anchor: "middle",
      halo: 6
    }, txt);
  }));
}

// ── the gap: every upper bound stops short of the line ────────────────────
function AV_Gap({
  topHi,
  th,
  xb,
  p,
  L,
  span
}) {
  if (p <= 0.004) return null;
  const a = xb(topHi),
    b = xb(th),
    yb = AV_BRK;
  return /*#__PURE__*/React.createElement("g", {
    opacity: p
  }, /*#__PURE__*/React.createElement("rect", {
    x: a,
    y: AV_ROW0 - 60,
    width: Math.max(0, b - a),
    height: AV_AXY - (AV_ROW0 - 60),
    fill: AV_FAINT
  }), /*#__PURE__*/React.createElement("line", {
    x1: a,
    y1: AV_ROW0 - 60,
    x2: a,
    y2: AV_AXY,
    stroke: AV_INK,
    strokeWidth: "2",
    strokeDasharray: "7 7",
    opacity: "0.55"
  }), /*#__PURE__*/React.createElement("line", {
    x1: a,
    y1: yb,
    x2: b,
    y2: yb,
    stroke: AV_INK,
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("line", {
    x1: a,
    y1: yb - 8,
    x2: a,
    y2: yb + 8,
    stroke: AV_INK,
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("line", {
    x1: b,
    y1: yb - 8,
    x2: b,
    y2: yb + 8,
    stroke: AV_CRIM,
    strokeWidth: "2.5"
  }), L && /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_R,
    y: AV_BAND,
    size: 22,
    weight: 600,
    anchor: "end"
  }, 'no interval reaches ' + avSig(th, 2)), L && /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_R,
    y: AV_SUB,
    size: 19,
    color: AV_GREY,
    anchor: "end"
  }, 'best upper bound ' + avSig(topHi, 3)));
}

// ── the held arm: the conclusive negative, alone, entirely below zero ────
// The control block is editorial, not computed: it states a result reported in
// reports/stage8_foldx.md (R2) and on the project site's main-result note —
// the seq_only control through the identical pipeline beats the structural arm
// at every L2 value tested, on validation. No number is invented for it.
const AV_CTRL = ['A sequence-only control', 'through the identical', 'pipeline beats it at every', 'L2 value tested.'];
const AV_CTRL2 = ['The loss is the structural', 'features, not the tuning.'];
function AV_Hold({
  arm,
  p,
  c,
  xb,
  y
}) {
  if (!arm || p <= 0.004 || !arm.hasCI) return null;
  const px = AV_CARD.x,
    py = AV_CARD.y;
  const pw = lerp(AV_CARD.w, AV_CARD.w2, c),
    ph = lerp(AV_CARD.h, AV_CARD.h2, c);
  const cx = (xb(arm.lo) + xb(arm.hi)) / 2;
  const tx = px + 26,
    rx = px + 428;
  const cp = clamp((c - 0.4) / 0.5, 0, 1);
  return /*#__PURE__*/React.createElement("g", {
    opacity: p
  }, /*#__PURE__*/React.createElement("polyline", {
    points: `${px + 24},${py + ph} ${px + 24},${y} ${(cx - (cx - px) / 2).toFixed(0)},${y}`,
    fill: "none",
    stroke: AV_CRIM,
    strokeWidth: "1.5",
    strokeDasharray: "6 6",
    opacity: "0.75"
  }), /*#__PURE__*/React.createElement("rect", {
    x: px,
    y: py,
    width: pw,
    height: ph,
    fill: AV_PANEL,
    stroke: AV_INK,
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("rect", {
    x: px,
    y: py,
    width: "6",
    height: ph,
    fill: AV_CRIM
  }), /*#__PURE__*/React.createElement(AV_Tx, {
    x: tx,
    y: py + 36,
    size: 17,
    weight: 600,
    color: AV_GREY,
    track: "0.16em"
  }, arm.cls === 'worse' ? 'HELD · CONCLUSIVE NEGATIVE' : 'HELD'), /*#__PURE__*/React.createElement(AV_Tx, {
    x: tx,
    y: py + 76,
    size: 21,
    weight: 600
  }, arm.name), /*#__PURE__*/React.createElement(AV_Tx, {
    x: tx,
    y: py + 140,
    size: 46,
    weight: 600,
    color: AV_CRIM
  }, 'Δ ' + avSig(arm.delta, 3)), /*#__PURE__*/React.createElement(AV_Tx, {
    x: tx,
    y: py + 176,
    size: 21,
    color: AV_INK
  }, '95% CI [ ' + avFix(arm.lo, 3) + '  ' + avFix(arm.hi, 3) + ' ]'), /*#__PURE__*/React.createElement(AV_Tx, {
    x: tx,
    y: py + 210,
    size: 19,
    weight: 600,
    color: AV_CRIM
  }, avSentence(arm.verdict)), cp > 0.004 && /*#__PURE__*/React.createElement("g", {
    opacity: cp
  }, /*#__PURE__*/React.createElement("line", {
    x1: px + 404,
    y1: py + 24,
    x2: px + 404,
    y2: py + ph - 24,
    stroke: AV_INK,
    strokeWidth: "1",
    opacity: "0.3"
  }), /*#__PURE__*/React.createElement(AV_Tx, {
    x: rx,
    y: py + 36,
    size: 14,
    weight: 600,
    color: AV_GREY,
    track: "0.14em"
  }, "CONTROL \xB7 IDENTICAL PIPELINE"), /*#__PURE__*/React.createElement(AV_Tx, {
    x: rx,
    y: py + 56,
    size: 13,
    weight: 500,
    color: AV_GREY,
    track: "0.14em"
  }, "STAGE 5 VALIDATION"), AV_CTRL.map((ln, k) => /*#__PURE__*/React.createElement(AV_Tx, {
    key: 'c' + k,
    x: rx,
    y: py + 96 + k * 26,
    size: 17
  }, ln)), AV_CTRL2.map((ln, k) => /*#__PURE__*/React.createElement(AV_Tx, {
    key: 'd' + k,
    x: rx,
    y: py + 216 + k * 26,
    size: 17,
    weight: 600,
    color: AV_CRIM
  }, ln))));
}

// ── caption, in act 4's block ─────────────────────────────────────────────
function AV_Caption({
  t,
  items
}) {
  let cur = null;
  for (let i = 0; i < items.length; i++) if (t >= items[i].at) cur = items[i];
  if (!cur) return null;
  const op = clamp((t - cur.at) / 0.4, 0, 1);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: AV_L,
      right: 380,
      top: 898,
      opacity: op,
      font: `400 32px ${AV_FONT}`,
      color: AV_INK,
      lineHeight: 1.34
    }
  }, /*#__PURE__*/React.createElement("div", null, cur.text, cur.accent ? /*#__PURE__*/React.createElement("span", {
    style: {
      color: AV_CRIM
    }
  }, " ", cur.accent) : null));
}

// ── the act ───────────────────────────────────────────────────────────────
function ActVerdict({
  T,
  t0,
  L
}) {
  const raw = T - (t0 || 0);
  if (raw < -0.3) return null;
  const t = Math.min(raw, AV_DUR); // hold the final frame
  const D = avData();
  if (!D) return null; // no data block — render nothing
  const arms = D.arms,
    th = D.th,
    n = arms.length;
  const labels = L !== false;

  // ── geometry derived from the numbers, so the layout cannot be wrong ────
  const pitch = Math.min(AV_PITCH, 460 / Math.max(1, n - 1));
  const rowY = i => AV_ROW0 + i * pitch;
  const rowsMid = AV_ROW0 + pitch * (n - 1) / 2;
  const thSpan = pitch * (n - 1) + 120;
  let amax = 0;
  arms.forEach(a => {
    amax = Math.max(amax, a.score, a.hasCI ? a.score + (a.hi - a.delta) : a.score);
  });
  amax = Math.max(0.1, Math.ceil(amax * 1.07 / 0.05) * 0.05);
  const xa = v => AV_X0 + v / amax * (AV_X1 - AV_X0);
  let dlo = 0,
    dhi = th;
  arms.forEach(a => {
    if (!a.hasCI) return;
    dlo = Math.min(dlo, a.lo, a.delta);
    dhi = Math.max(dhi, a.hi, a.delta);
  });
  const dpad = 0.07 * Math.max(1e-6, dhi - dlo);
  const dmin = dlo - dpad,
    dmax = dhi + dpad * 2.4;
  const xb = v => AV_X0 + (v - dmin) / (dmax - dmin) * (AV_X1 - AV_X0);

  // the arm the act ends on, found by its verdict and not by name: the
  // strongest conclusive negative, falling back to the lowest-scoring arm
  let hi = -1,
    worst = Infinity;
  arms.forEach((a, i) => {
    if (a.cls === 'worse' && a.delta < worst) {
      worst = a.delta;
      hi = i;
    }
  });
  if (hi < 0) hi = arms.length - 1;
  const held = arms[hi];
  // the highest upper bound anywhere: the gap beat only claims what is true
  let topHi = -Infinity;
  arms.forEach(a => {
    if (a.hasCI) topHi = Math.max(topHi, a.hi);
  });
  const showGap = isFinite(topHi) && topHi < th;

  // ── timing ──────────────────────────────────────────────────────────────
  const fade = tw(t, 0, 0.18, AV_LIN);
  const m = tw(t, AVT.morph, AVT.morphE, AV_IO); // bars → deltas
  const holdP = tw(t, AVT.dim, AVT.dim + 0.5, AV_LIN);
  const panelP = tw(t, AVT.hold, AVT.hold + 0.45, AV_OUT);
  const rowsOut = 1; // the rows stay until the last frame
  const thOp = fade; // so does the rule
  const gapP = showGap ? tw(t, AVT.gap, AVT.gap + 0.6, AV_IO) * lerp(1, 0.2, holdP) : 0;
  // the frame builds as the rule docks; before that the rule is alone
  const frameP = tw(t, AVT.dock - 0.15, AVT.dock + 0.65, AV_IO);
  const declOut = (1 - tw(t, AVT.dock - 0.40, AVT.dock + 0.20, AV_IO)) * fade;
  // the control finding pulls focus onto the held arm alone
  const ctrlP = tw(t, AVT.ctrl, AVT.ctrl + 0.6, AV_IO);
  const dimOf = i => i === hi ? 1 : lerp(1, 0.14, holdP) * (1 - ctrlP);
  const emphOf = i => i === hi ? holdP : 0;
  const textOf = i => i === hi ? 1 : 1 - holdP;
  // each row's interval swells as its own verdict lands
  const pulseOf = i => {
    const q = (t - (AVT.vrd + i * AVT.vstep) + 0.12) / 0.9;
    return q > 0 && q < 1 ? Math.sin(Math.PI * q) : 0;
  };
  const g = avThreshGeom(t, th, xb, rowsMid, thSpan);

  // the scoring context, from the payload; the resample count only if present
  let ctx = 'TEST SPLIT';
  if (isFinite(arms[0].nRows)) ctx += ' · ' + avGrp(arms[0].nRows) + ' ROWS';
  if (isFinite(arms[0].nAll)) ctx += ' · ' + avGrp(arms[0].nAll) + ' ALLELES';
  ctx += ' · PAIRED CLUSTER BOOTSTRAP';
  if (isFinite(D.boot)) ctx += ' · ' + avGrp(D.boot) + ' RESAMPLES';
  const caps = [{
    at: 0.30,
    text: 'The rules were fixed before any modelling.'
  }, {
    at: 4.50,
    text: 'Nothing had been scored when this bar was set.'
  }, {
    at: AVT.bars,
    text: 'Six arms. One test set.',
    accent: 'Scored once.'
  }, {
    at: 8.30,
    text: 'Each bar carries its paired 95% interval.'
  }, {
    at: AVT.morphE + 0.15,
    text: 'Redrawn as a difference from the baseline sequence ensemble.'
  }, {
    at: 12.90,
    text: 'The ESM-2 arms neither replace nor improve the baseline.'
  }, {
    at: AVT.gap + 0.4,
    text: 'Every upper bound stops short of the line.',
    accent: 'No arm earns the switch.'
  }, {
    at: AVT.hold + 0.2,
    text: 'The structural arm is not merely no better —',
    accent: 'its whole interval is below zero.'
  }, {
    at: AVT.ctrl + 0.25,
    text: 'Not a tuning artefact —',
    accent: 'the loss is the structural features.'
  }];
  const headA = (1 - m) * fade * frameP,
    headB = m * fade;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      background: AV_BG,
      fontFamily: AV_FONT,
      opacity: fade
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "1920",
    height: "1080",
    style: {
      position: 'absolute',
      inset: 0
    }
  }, /*#__PURE__*/React.createElement("line", {
    x1: AV_L,
    y1: "150",
    x2: AV_R,
    y2: "150",
    stroke: AV_INK,
    strokeWidth: "1",
    opacity: 0.25 * fade
  }), /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_L,
    y: AV_BAND,
    op: headA,
    size: 18,
    color: AV_GREY,
    track: "0.18em"
  }, 'TEST ρ · MEDIAN PER ALLELE · ' + n + ' ARMS'), /*#__PURE__*/React.createElement(AV_Tx, {
    x: AV_L,
    y: AV_BAND,
    op: headB,
    size: 18,
    color: AV_GREY,
    track: "0.18em"
  }, 'Δ ρ VS THE BASELINE · 95% PAIRED CI'), /*#__PURE__*/React.createElement(AV_Axis, {
    m: m,
    amax: amax,
    xa: xa,
    dmin: dmin,
    dmax: dmax,
    xb: xb,
    frameP: frameP
  }), labels && /*#__PURE__*/React.createElement(AV_Declare, {
    t: t,
    th: th,
    out: declOut,
    ctx: ctx
  }), /*#__PURE__*/React.createElement(AV_Gap, {
    topHi: topHi,
    th: th,
    xb: xb,
    p: gapP,
    L: labels,
    span: thSpan
  }), /*#__PURE__*/React.createElement(AV_Threshold, {
    t: t,
    th: th,
    g: g,
    op: thOp
  }), /*#__PURE__*/React.createElement(AV_Rows, {
    t: t,
    arms: arms,
    th: th,
    m: m,
    xa: xa,
    xb: xb,
    rowY: rowY,
    out: rowsOut,
    dimOf: dimOf,
    emphOf: emphOf,
    textOf: textOf,
    pulseOf: pulseOf
  }), labels && /*#__PURE__*/React.createElement(AV_Verdicts, {
    t: t,
    arms: arms,
    xb: xb,
    rowY: rowY,
    out: rowsOut,
    dimOf: dimOf,
    textOf: textOf
  }), labels && held && /*#__PURE__*/React.createElement(AV_Hold, {
    arm: held,
    p: panelP * rowsOut,
    c: ctrlP,
    xb: xb,
    y: rowY(hi)
  })), labels && /*#__PURE__*/React.createElement(AV_Caption, {
    t: t,
    items: caps
  }));
}
window.ActVerdict = ActVerdict;
;
/* src/animations/submission-film/act6-transfer.jsx */
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
  axes: 0.3,
  title: 0.55,
  head: 0.7,
  body: 1.15,
  curve: 1.05,
  curveEnd: 3.05,
  // ROC draws; the AUROC counts up with it
  iqr: 2.95,
  event: 3.3,
  auprc: 3.6,
  tile: [4.05, 4.2, 4.35],
  caveat: 4.6,
  swap: 5.0,
  // beat 1 column out, ladder in
  ladder: 5.2,
  // THE ARGUMENT, one control at a time, 2.7 s apart. Each lands its bar, its
  // own ROC in the left panel, then its verdict — so the three read as a
  // sequence of claims rather than as a table.
  row: [5.7, 8.4, 11.1],
  // random · other-allele · wrong groove
  verdict: [6.7, 9.4, 13.4],
  collapse: 12.5,
  collapseEnd: 13.15,
  // the wrong row is built at the primary
  delta: 13.35,
  // value, held there, and only then falls
  caption: 13.8,
  claim: 14.9
};

// ── geometry, stage px on 1920×1080 ─────────────────────────────────────────
const AT6_P = {
  x0: 200,
  x1: 760,
  y0: 860,
  y1: 300
}; // ROC plot box
const AT6_RX = 880; // right column left edge
const AT6_LAD = {
  x0: 900,
  x1: 1780,
  axisY: 430,
  rowY: [500, 630, 760]
};
const AT6_X = f => AT6_P.x0 + f * (AT6_P.x1 - AT6_P.x0);
const AT6_Y = t => AT6_P.y0 - t * (AT6_P.y0 - AT6_P.y1);
// AUROC → ladder x. 0.5 (chance) is the origin; 1.0 is the right edge.
const AT6_LX = v => AT6_LAD.x0 + Math.min(1, Math.max(0, (v - 0.5) / 0.5)) * (AT6_LAD.x1 - AT6_LAD.x0);

// ── data, resolved at render time ───────────────────────────────────────────
function AT6_data() {
  const F = typeof window === 'undefined' ? null : window.__FILM__;
  const E = F && F.elution;
  if (!E || typeof E !== 'object') return null;
  const n = v => typeof v === 'number' && isFinite(v) ? v : null;
  const ctl = k => {
    const o = E.controls && E.controls[k];
    if (o == null) return null;
    return n(typeof o === 'number' ? o : o.auroc_median);
  };
  const primary = n(E.auroc_median);
  if (primary == null) return null; // nothing to draw: render nothing
  return {
    primary,
    iqrLo: n(E.auroc_iqr_low),
    iqrHi: n(E.auroc_iqr_high),
    auprc: n(E.auprc_median),
    auprcChance: n(E.auprc_chance),
    nAlleles: n(E.n_alleles),
    nPeptides: n(E.n_peptides),
    nDecoys: n(E.n_decoys),
    ratio: n(E.decoy_ratio),
    random: ctl('random_scores'),
    other: ctl('other_allele_ligand_decoys'),
    wrong: ctl('wrong_hla_pseudosequence')
  };
}

// display-only rounding
const AT6_f3 = v => v == null ? '—' : v.toFixed(3);
const AT6_int = v => v == null ? '—' : Math.round(v).toLocaleString('en-US');

// tpr = fpr^((1-A)/A): the one-parameter ROC whose area is exactly A.
function AT6_roc(a) {
  const A = Math.min(0.9995, Math.max(0.5001, a));
  const k = (1 - A) / A;
  let d = '';
  for (let i = 0; i <= 180; i++) {
    const u = i / 180,
      f = u * u * u; // dense sampling near fpr = 0
    d += (i === 0 ? 'M' : 'L') + AT6_X(f).toFixed(2) + ' ' + AT6_Y(Math.pow(f, k)).toFixed(2);
  }
  return d;
}

// ── small type helper: all type is HTML, all marks are SVG ──────────────────
function AT6Txt({
  x,
  y,
  op,
  size,
  weight,
  color,
  align,
  ls,
  lh,
  width,
  wrap,
  children
}) {
  if (op != null && op <= 0.001) return null;
  const tx = align === 'right' ? '-100%' : align === 'center' ? '-50%' : '0';
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: x,
      top: y,
      transform: `translate(${tx},0)`,
      opacity: op == null ? 1 : op,
      font: `${weight || 500} ${size || 24}px ${FONT}`,
      color: color || C.ink,
      letterSpacing: ls,
      lineHeight: lh,
      width: width,
      whiteSpace: wrap ? 'normal' : 'nowrap'
    }
  }, children);
}
function AT6Wash({
  op
}) {
  if (op <= 0.001) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: op,
      background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)'
    }
  });
}

// ── beat 1 + 2 share the ROC panel on the left ──────────────────────────────
function AT6Plot({
  t,
  d,
  L,
  drawP,
  ghosts
}) {
  const ax = tw(t, AT6_T.axes, AT6_T.axes + 0.55, MOTION.draw);
  const gridOp = tw(t, AT6_T.axes + 0.2, AT6_T.axes + 0.7, Easing.linear);
  const chance = tw(t, AT6_T.axes + 0.35, AT6_T.axes + 0.9, Easing.linear);
  const w = AT6_P.x1 - AT6_P.x0,
    h = AT6_P.y0 - AT6_P.y1;
  return /*#__PURE__*/React.createElement("g", null, [0.25, 0.5, 0.75].map(g => /*#__PURE__*/React.createElement("g", {
    key: g,
    opacity: gridOp * 0.9
  }, /*#__PURE__*/React.createElement("line", {
    x1: AT6_X(g),
    y1: AT6_P.y0,
    x2: AT6_X(g),
    y2: AT6_P.y1,
    stroke: "#C9CCD4",
    strokeWidth: "1"
  }), /*#__PURE__*/React.createElement("line", {
    x1: AT6_P.x0,
    y1: AT6_Y(g),
    x2: AT6_P.x1,
    y2: AT6_Y(g),
    stroke: "#C9CCD4",
    strokeWidth: "1"
  }))), /*#__PURE__*/React.createElement("line", {
    x1: AT6_P.x0,
    y1: AT6_P.y0,
    x2: AT6_P.x0,
    y2: AT6_P.y0 - h * ax,
    stroke: C.ink,
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("line", {
    x1: AT6_P.x0,
    y1: AT6_P.y0,
    x2: AT6_P.x0 + w * ax,
    y2: AT6_P.y0,
    stroke: C.ink,
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("line", {
    x1: AT6_P.x0,
    y1: AT6_P.y0,
    x2: AT6_X(chance),
    y2: AT6_Y(chance),
    stroke: C.helix,
    strokeWidth: "2",
    strokeDasharray: "7 9",
    opacity: 0.85
  }), ghosts.map(g => /*#__PURE__*/React.createElement("path", {
    key: g.key,
    d: AT6_roc(g.v),
    fill: "none",
    stroke: g.color,
    strokeWidth: "2.5",
    strokeDasharray: g.dash,
    opacity: g.op,
    pathLength: "1"
  })), /*#__PURE__*/React.createElement("path", {
    d: AT6_roc(d.primary),
    fill: "none",
    stroke: C.ink,
    strokeWidth: "4.5",
    strokeLinecap: "round",
    pathLength: "1",
    strokeDasharray: "1",
    strokeDashoffset: 1 - drawP
  }), L && chance > 0.5 && /*#__PURE__*/React.createElement("circle", {
    cx: AT6_X(0.5),
    cy: AT6_Y(0.5),
    r: "4",
    fill: C.helix,
    opacity: chance
  }));
}

// ── beat 2: the control ladder ──────────────────────────────────────────────
function AT6_rows(d) {
  const rows = [];
  if (d.random != null) rows.push({
    key: 'random',
    v: d.random,
    color: C.helix,
    dash: '6 7',
    label: 'random scores',
    note: 'the scoring harness itself, fed noise',
    verdict: 'chance — the floor'
  });
  if (d.other != null) rows.push({
    key: 'other',
    v: d.other,
    color: '#1D3557',
    dash: null,
    label: "other alleles' ligands as decoys",
    note: 'negatives that are themselves presented',
    verdict: 'not just "looks presentable"'
  });
  if (d.wrong != null) rows.push({
    key: 'wrong',
    v: d.wrong,
    color: C.crimson,
    dash: null,
    collapse: true,
    label: 'the WRONG HLA pseudosequence',
    note: 'same peptides, a different groove'
    // this row's verdict is the measured drop, drawn by the delta below
  });
  return rows;
}
function AT6Ladder({
  t,
  d,
  L,
  op,
  rows
}) {
  if (op <= 0.001) return null;
  const px = AT6_LX(d.primary);
  const axis = tw(t, AT6_T.ladder, AT6_T.ladder + 0.5, MOTION.draw);
  return /*#__PURE__*/React.createElement("g", {
    opacity: op
  }, /*#__PURE__*/React.createElement("line", {
    x1: AT6_LAD.x0,
    y1: AT6_LAD.axisY,
    x2: AT6_LAD.x0 + (AT6_LAD.x1 - AT6_LAD.x0) * axis,
    y2: AT6_LAD.axisY,
    stroke: C.ink,
    strokeWidth: "2"
  }), [0.5, 0.75, 1].map(v => /*#__PURE__*/React.createElement("line", {
    key: v,
    x1: AT6_LX(v),
    y1: AT6_LAD.axisY,
    x2: AT6_LX(v),
    y2: AT6_LAD.axisY - 9,
    stroke: C.ink,
    strokeWidth: "2",
    opacity: axis
  })), /*#__PURE__*/React.createElement("line", {
    x1: px,
    y1: AT6_LAD.axisY,
    x2: px,
    y2: AT6_LAD.rowY[2] + 56,
    stroke: C.ink,
    strokeWidth: "1.5",
    strokeDasharray: "5 7",
    opacity: 0.42 * axis
  }), rows.map((r, i) => {
    const y = AT6_LAD.rowY[i];
    const p = tw(t, AT6_T.row[i], AT6_T.row[i] + 0.45, MOTION.draw);
    if (p <= 0.001) return null;
    const v = r.shown;
    const x = lerp(AT6_LAD.x0, AT6_LX(v), p);
    return /*#__PURE__*/React.createElement("g", {
      key: r.key
    }, /*#__PURE__*/React.createElement("rect", {
      x: AT6_LAD.x0,
      y: y,
      width: Math.max(0, x - AT6_LAD.x0),
      height: "12",
      fill: r.color
    }), /*#__PURE__*/React.createElement("line", {
      x1: x,
      y1: y - 9,
      x2: x,
      y2: y + 21,
      stroke: C.ink,
      strokeWidth: "2.5"
    }), /*#__PURE__*/React.createElement("rect", {
      x: AT6_RX,
      y: y - 48,
      width: "14",
      height: "14",
      fill: r.color,
      stroke: C.ink,
      strokeWidth: "1.5"
    }));
  }), rows.map((r, i) => {
    if (!r.collapse || r.deltaOp <= 0.001) return null;
    const y = AT6_LAD.rowY[i] + 6;
    const a = AT6_LX(r.shown),
      b = px;
    return /*#__PURE__*/React.createElement("g", {
      key: 'd' + r.key,
      opacity: r.deltaOp
    }, /*#__PURE__*/React.createElement("line", {
      x1: a,
      y1: y,
      x2: b,
      y2: y,
      stroke: C.crimson,
      strokeWidth: "2.5",
      strokeDasharray: "7 7"
    }), /*#__PURE__*/React.createElement("line", {
      x1: b,
      y1: y - 12,
      x2: b,
      y2: y + 12,
      stroke: C.crimson,
      strokeWidth: "2.5"
    }));
  }));
}

// ── the act ─────────────────────────────────────────────────────────────────
function ActTransfer({
  T,
  t0,
  L
}) {
  const t = T - (t0 || 0);
  if (t < -0.3 || t > 16.6) return null;
  const d = AT6_data();
  if (!d) return null; // no data file: render nothing
  const labels = L !== false;
  const appear = tw(t, 0, 0.3, Easing.linear);
  const beats = appear; // nothing fades out: ActNext dissolves over this
  const col1 = 1 - tw(t, AT6_T.swap, AT6_T.swap + 0.3, Easing.linear); // beat 1 column
  const col2 = tw(t, AT6_T.swap + 0.2, AT6_T.swap + 0.55, Easing.linear); // beat 2 ladder

  // the curve draws and the headline number counts up on one ramp
  const drawP = tw(t, AT6_T.curve, AT6_T.curveEnd, Easing.easeOutCubic);
  const shownAuroc = lerp(0.5, d.primary, drawP);

  // the collapse: the wrong-groove row is built at the primary value, then
  // falls to its own. Both ends come from the data, so the drop is measured.
  const collapseP = tw(t, AT6_T.collapse, AT6_T.collapseEnd, Easing.easeInOutQuart);
  const rows = AT6_rows(d).map(r => {
    if (!r.collapse) return Object.assign({}, r, {
      shown: r.v,
      deltaOp: 0
    });
    return Object.assign({}, r, {
      shown: lerp(d.primary, r.v, collapseP),
      deltaOp: tw(t, AT6_T.delta, AT6_T.delta + 0.3, Easing.linear)
    });
  });

  // each control also draws its ROC into the left panel as it lands
  const ghosts = rows.map((r, i) => ({
    // the random control's ROC *is* the chance diagonal, so it would draw a
    // second line on top of one already there; the labelled diagonal carries it.
    key: r.key,
    v: r.shown,
    color: r.color,
    dash: r.dash,
    ghost: r.key !== 'random',
    op: tw(t, AT6_T.row[i], AT6_T.row[i] + 0.5, Easing.linear) * col2 * 0.95
  })).filter(g => g.ghost && g.op > 0.001);
  const wrongRow = rows.filter(r => r.collapse)[0] || null;
  const drop = wrongRow ? d.primary - wrongRow.v : null;
  const fold = d.auprc != null && d.auprcChance ? d.auprc / d.auprcChance : null;
  const tiles = [[AT6_int(d.nPeptides), 'eluted ligands'], [AT6_int(d.nDecoys), 'proteome decoys'], [AT6_int(d.nAlleles), 'HLA alleles']];
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      fontFamily: FONT
    }
  }, /*#__PURE__*/React.createElement(AT6Wash, {
    op: appear
  }), beats > 0.001 && /*#__PURE__*/React.createElement("svg", {
    width: "1920",
    height: "1080",
    style: {
      position: 'absolute',
      inset: 0
    },
    opacity: beats
  }, /*#__PURE__*/React.createElement(AT6Plot, {
    t: t,
    d: d,
    L: labels,
    drawP: drawP,
    ghosts: ghosts
  }), /*#__PURE__*/React.createElement(AT6Ladder, {
    t: t,
    d: d,
    L: labels,
    op: col2,
    rows: rows
  }), col1 > 0.001 && d.auprc != null && d.auprcChance != null && (() => {
    const p = tw(t, AT6_T.auprc, AT6_T.auprc + 0.55, MOTION.draw) * col1;
    if (p <= 0.001) return null;
    const bx = AT6_RX + 170,
      bw = 500;
    return /*#__PURE__*/React.createElement("g", {
      opacity: col1
    }, /*#__PURE__*/React.createElement("rect", {
      x: bx,
      y: "802",
      width: bw * d.auprc * p,
      height: "11",
      fill: C.ink
    }), /*#__PURE__*/React.createElement("rect", {
      x: bx,
      y: "832",
      width: Math.max(2, bw * d.auprcChance * p),
      height: "11",
      fill: C.helix
    }), /*#__PURE__*/React.createElement("line", {
      x1: bx,
      y1: "794",
      x2: bx,
      y2: "851",
      stroke: C.ink,
      strokeWidth: "1.5",
      opacity: 0.5 * p
    }));
  })()), beats > 0.001 && col1 > 0.001 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: beats * col1
    }
  }, /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_P.x0,
    y: 248,
    op: tw(t, AT6_T.title, AT6_T.title + 0.4, Easing.linear),
    size: 20,
    weight: 600,
    ls: "0.13em",
    color: "#5A5C6B"
  }, "ELUTED LIGANDS vs PROTEOME DECOYS"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 296,
    op: tw(t, AT6_T.head - 0.1, AT6_T.head + 0.3, Easing.linear),
    size: 20,
    weight: 600,
    ls: "0.15em",
    color: "#5A5C6B"
  }, "A SECOND ASSAY \xB7 NO RETRAINING"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 352,
    op: tw(t, AT6_T.head, AT6_T.head + 0.5, MOTION.enter),
    size: 46,
    weight: 600
  }, "Nobody asked it for this."), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 424,
    op: tw(t, AT6_T.body, AT6_T.body + 0.5, Easing.linear),
    size: 29,
    weight: 400,
    lh: "1.5",
    width: 900,
    wrap: true,
    color: "#3C3E52"
  }, "Trained on dissociation half-life alone and never retrained, it ranked peptides recovered from living cells above matched decoys."), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 576,
    op: tw(t, AT6_T.curve, AT6_T.curve + 0.4, Easing.linear),
    size: 108,
    weight: 600,
    lh: "1"
  }, AT6_f3(shownAuroc)), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 702,
    op: tw(t, AT6_T.iqr, AT6_T.iqr + 0.4, Easing.linear),
    size: 24,
    color: "#3C3E52"
  }, "median AUROC", d.nAlleles != null ? ` · ${AT6_int(d.nAlleles)} alleles` : '', d.iqrLo != null && d.iqrHi != null ? ` · IQR ${AT6_f3(d.iqrLo)}–${AT6_f3(d.iqrHi)}` : ''), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 736,
    op: tw(t, AT6_T.event, AT6_T.event + 0.4, Easing.linear),
    size: 24,
    color: "#6A6C7A"
  }, "a different assay \xB7 a different biological event"), d.auprc != null && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 796,
    op: tw(t, AT6_T.auprc, AT6_T.auprc + 0.4, Easing.linear),
    size: 21,
    color: "#3C3E52"
  }, "AUPRC ", AT6_f3(d.auprc)), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 826,
    op: tw(t, AT6_T.auprc + 0.1, AT6_T.auprc + 0.5, Easing.linear),
    size: 21,
    color: "#6A6C7A"
  }, "chance ", AT6_f3(d.auprcChance)), fold != null && /*#__PURE__*/React.createElement(AT6Txt, {
    x: 1800,
    y: 808,
    align: "right",
    op: tw(t, AT6_T.auprc + 0.25, AT6_T.auprc + 0.65, Easing.linear),
    size: 26,
    weight: 600
  }, fold.toFixed(1), "\xD7 chance", d.ratio ? ` at ${Math.round(d.ratio)}:1` : '')), tiles.map((tile, i) => /*#__PURE__*/React.createElement(React.Fragment, {
    key: i
  }, /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX + i * 310,
    y: 888,
    op: tw(t, AT6_T.tile[i], AT6_T.tile[i] + 0.35, MOTION.enter),
    size: 42,
    weight: 600
  }, tile[0]), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX + i * 310,
    y: 940,
    op: tw(t, AT6_T.tile[i] + 0.08, AT6_T.tile[i] + 0.4, Easing.linear),
    size: 20,
    color: "#6A6C7A"
  }, tile[1]))), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 992,
    op: tw(t, AT6_T.caveat, AT6_T.caveat + 0.4, Easing.linear),
    size: 20,
    weight: 400,
    color: "#6A6C7A"
  }, "Transfer to a different assay \u2014 not a second measurement of stability accuracy.")), beats > 0.001 && col2 > 0.001 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: beats * col2
    }
  }, /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 296,
    size: 20,
    weight: 600,
    ls: "0.15em",
    color: "#5A5C6B"
  }, "THE CONTROLS"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 336,
    size: 34,
    weight: 600
  }, "Three ways this could have been fake."), labels && [0.5, 0.75, 1].map(v => /*#__PURE__*/React.createElement(AT6Txt, {
    key: v,
    x: AT6_LX(v),
    y: 386,
    align: "center",
    size: 19,
    color: "#6A6C7A"
  }, v.toFixed(2))), rows.map((r, i) => {
    const p = tw(t, AT6_T.row[i], AT6_T.row[i] + 0.4, Easing.linear);
    if (p <= 0.001) return null;
    const y = AT6_LAD.rowY[i];
    return /*#__PURE__*/React.createElement(React.Fragment, {
      key: r.key
    }, /*#__PURE__*/React.createElement(AT6Txt, {
      x: AT6_RX + 26,
      y: y - 52,
      op: p,
      size: 26,
      weight: 600
    }, r.label), /*#__PURE__*/React.createElement(AT6Txt, {
      x: AT6_LAD.x1,
      y: y - 56,
      align: "right",
      op: p,
      size: 30,
      weight: 600,
      color: r.collapse ? C.crimson : C.ink
    }, AT6_f3(r.shown)), labels && /*#__PURE__*/React.createElement(AT6Txt, {
      x: AT6_RX + 26,
      y: y + 30,
      op: p,
      size: 19,
      color: "#6A6C7A"
    }, r.note), r.verdict && /*#__PURE__*/React.createElement(AT6Txt, {
      x: AT6_LAD.x1,
      y: y + 26,
      align: "right",
      size: 22,
      weight: 600,
      op: tw(t, AT6_T.verdict[i], AT6_T.verdict[i] + 0.4, Easing.linear)
    }, r.verdict));
  }), wrongRow && drop != null && /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_LX(d.primary) - 16,
    y: AT6_LAD.rowY[2] + 26,
    align: "right",
    op: wrongRow.deltaOp,
    size: 24,
    weight: 600,
    color: C.crimson
  }, "\u2212", drop.toFixed(3), " AUROC"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_LAD.x1,
    y: AT6_LAD.rowY[2] + 62,
    align: "right",
    size: 19,
    color: "#6A6C7A"
  }, "primary ", AT6_f3(d.primary)), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 872,
    op: tw(t, AT6_T.caption, AT6_T.caption + 0.4, MOTION.enter),
    size: 34,
    weight: 600
  }, "It did not memorise peptides."), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 916,
    op: tw(t, AT6_T.caption + 0.18, AT6_T.caption + 0.58, MOTION.enter),
    size: 34,
    weight: 600
  }, "It learned the allele."), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_RX,
    y: 972,
    op: tw(t, AT6_T.claim, AT6_T.claim + 0.5, Easing.linear),
    size: 25,
    color: "#3C3E52"
  }, "Evidence it learned real biology \u2014 not dataset artefacts."), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_P.x0,
    y: 930,
    op: tw(t, AT6_T.ladder + 0.15, AT6_T.ladder + 0.55, Easing.linear),
    size: 20,
    color: "#6A6C7A"
  }, AT6_int(d.nPeptides), " ligands \xB7 ", AT6_int(d.nDecoys), " decoys", d.ratio ? ` · ${Math.round(d.ratio)}:1` : '')), beats > 0.001 && labels && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: beats
    }
  }, /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_P.x0,
    y: AT6_P.y0 + 12,
    align: "center",
    size: 19,
    color: "#6A6C7A",
    op: tw(t, AT6_T.axes + 0.3, AT6_T.axes + 0.7, Easing.linear)
  }, "0"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_P.x1,
    y: AT6_P.y0 + 12,
    align: "center",
    size: 19,
    color: "#6A6C7A",
    op: tw(t, AT6_T.axes + 0.3, AT6_T.axes + 0.7, Easing.linear)
  }, "1"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_P.x0 - 18,
    y: AT6_P.y1 - 12,
    align: "right",
    size: 19,
    color: "#6A6C7A",
    op: tw(t, AT6_T.axes + 0.3, AT6_T.axes + 0.7, Easing.linear)
  }, "1"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_X(0.80) + 10,
    y: AT6_Y(0.80) + 4,
    size: 18,
    color: "#8A8B90",
    op: tw(t, AT6_T.axes + 0.5, AT6_T.axes + 0.9, Easing.linear)
  }, "chance"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: (AT6_P.x0 + AT6_P.x1) / 2,
    y: AT6_P.y0 + 42,
    align: "center",
    size: 21,
    color: "#3C3E52",
    op: tw(t, AT6_T.axes + 0.4, AT6_T.axes + 0.8, Easing.linear)
  }, "false positive rate"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: AT6_P.x0 - 96,
      top: (AT6_P.y0 + AT6_P.y1) / 2,
      transform: 'translate(-50%,-50%) rotate(-90deg)',
      font: `500 21px ${FONT}`,
      color: '#3C3E52',
      whiteSpace: 'nowrap',
      opacity: tw(t, AT6_T.axes + 0.4, AT6_T.axes + 0.8, Easing.linear)
    }
  }, "true positive rate"), /*#__PURE__*/React.createElement(AT6Txt, {
    x: AT6_P.x0,
    y: 930,
    size: 18,
    color: "#8A8B90",
    op: tw(t, AT6_T.curveEnd, AT6_T.curveEnd + 0.5, Easing.linear) * col1
  }, "curve shape drawn to the measured AUROC; per-allele medians")));
}
window.ActTransfer = ActTransfer;
;
/* src/animations/submission-film/act7-next.jsx */
// act7-next.jsx — Act VII, "what we would do next" + the close (14 s).
//
// Not an ES module. Eval'd into one shared global scope after animations-v3.jsx
// and pmhc-scene-v2.jsx, so it borrows C, FONT, MOTION, tw, lerp, clamp and
// Easing by bare name and publishes window.ActNext. Every module-scope symbol
// is prefixed NX_ / NX so it can never collide with the intro scene or a
// sibling act — a duplicate top-level `const` in this loader is a hard
// SyntaxError, not a shadowing.
//
// THE BEAT. Act VI closed the evidence. This act spends nine seconds on the
// three experiments we would run next, then five on the close.
//
// WHY THE "vs." COLUMN IS LOAD-BEARING. A next-steps slide is a wishlist unless
// every row names the thing it would be measured against. Each row here carries
// the same motif — proposal on the left, the control it is compared to on the
// right — so a judge can see that all three are falsifiable before reading a
// word of the rationale. Do not drop the right column to buy screen time; drop
// a row instead.
//
// COPY IS HARDCODED AND THAT IS CORRECT. The three rows are transcribed from
// the "Next steps" table in site/index.html, and the credits from its footer.
// There is no data contract: none of these experiments has been run, so there
// is nothing in window.__FILM__ to read and nothing to count up. The act
// renders identically with or without the data file — never invent a result,
// an effect size or a score for any row here.
//
// THE BOOKEND. The film opens on a question (the site's title, "Sequence Is All
// You Need?"). The close poses it again and answers it with the thesis the
// whole film was built to earn. Those two ink lines are the point of the film;
// the credits are subordinate to them in size and in weight.

// ── timing, in act-local seconds ────────────────────────────────────────────
const NX_DUR = 14;
const NX_T = {
  wash: 0.0,
  eyebrow: 0.3,
  title: 0.5,
  rule: 0.9,
  row: [1.5, 3.4, 5.3],
  // one experiment every ~1.9 s
  foot: 7.4,
  out: 8.6,
  outEnd: 9.15,
  // the table leaves
  // the close
  ask: 9.3,
  question: 9.55,
  qrule: 10.05,
  thesis1: 10.35,
  thesis2: 10.95,
  crule: 11.55,
  credits: 11.85,
  credits2: 12.2,
  disclose: 12.5
};

// ── geometry, stage px on 1920×1080 ─────────────────────────────────────────
const NX_L = 190; // left margin / rule start
const NX_R = 1730; // right margin / rule end
const NX_TXT = 272; // title + rationale column
const NX_VS = 1380; // "vs." column left edge
const NX_ROWY = [336, 540, 744];
const NX_HEAD_RULE = 300;

// ── the three proposals, transcribed from site/index.html "Next steps" ──────
// `vs` is the named control; without it a row is not an experiment.
const NX_ROWS = [{
  key: 'esmc',
  n: '01',
  glyph: 'concat',
  title: 'ESM-C with concatenated sequences',
  body: 'Feed peptide–linker–MHC α1α2 into ESM-C as one sequence, so each partner is ' + 'represented in the context of the other.',
  vs: 'encoding peptide and MHC separately'
}, {
  key: 'xattn',
  n: '02',
  glyph: 'cross',
  title: 'Separate embeddings + cross-attention',
  body: 'Keep peptide and MHC embeddings separate, then add a trainable cross-attention ' + 'module linking peptide positions to groove positions.',
  vs: 'concatenated embeddings + a small MLP'
}, {
  key: 'energy',
  n: '03',
  glyph: 'graph',
  title: 'Inverse-folding and energy features',
  body: 'Score each predicted complex with ProteinMPNN peptide likelihoods and FoldX ' + 'interaction energies, then test them as half-life features.',
  vs: 'the sequence ensemble, same folds and tuning budget'
}];

// ── small type helper: all type is HTML, all marks are SVG ──────────────────
function NXTxt({
  x,
  y,
  op,
  size,
  weight,
  color,
  align,
  ls,
  lh,
  width,
  wrap,
  children
}) {
  if (op != null && op <= 0.001) return null;
  const tx = align === 'right' ? '-100%' : align === 'center' ? '-50%' : '0';
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: x,
      top: y,
      transform: `translate(${tx},0)`,
      opacity: op == null ? 1 : op,
      font: `${weight || 500} ${size || 24}px ${FONT}`,
      color: color || C.ink,
      letterSpacing: ls,
      lineHeight: lh,
      width: width,
      whiteSpace: wrap ? 'normal' : 'nowrap'
    }
  }, children);
}
function NXWash({
  op
}) {
  if (op <= 0.001) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: op,
      background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)'
    }
  });
}

// ── row glyphs: a 62 × 46 hairline box in the left gutter ─────────────────
// Each one is the architecture the row proposes, at a glance: one joined
// chain, two chains with trained links between them, a scored complex. The box
// is what keeps the marks from reading as stray typography next to the body
// copy — without it the "one sequence" bar looks like an em dash.
const NX_GW = 62,
  NX_GH = 46;
function NXGlyph({
  kind,
  x,
  y,
  op
}) {
  if (op <= 0.001) return null;
  const g = C.helix;
  let marks;
  if (kind === 'concat') {
    marks = /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("rect", {
      x: x + 9,
      y: y + 19,
      width: "15",
      height: "9",
      fill: C.ink
    }), /*#__PURE__*/React.createElement("line", {
      x1: x + 25,
      y1: y + 23.5,
      x2: x + 33,
      y2: y + 23.5,
      stroke: g,
      strokeWidth: "2",
      strokeDasharray: "2.5 2.5"
    }), /*#__PURE__*/React.createElement("rect", {
      x: x + 34,
      y: y + 19,
      width: "19",
      height: "9",
      fill: g
    }));
  } else if (kind === 'cross') {
    marks = /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("rect", {
      x: x + 9,
      y: y + 11,
      width: "26",
      height: "7",
      fill: C.ink
    }), /*#__PURE__*/React.createElement("rect", {
      x: x + 27,
      y: y + 29,
      width: "26",
      height: "7",
      fill: g
    }), [0, 1, 2].map(i => /*#__PURE__*/React.createElement("line", {
      key: i,
      x1: x + 13 + i * 9,
      y1: y + 19,
      x2: x + 31 + i * 9,
      y2: y + 28,
      stroke: C.ink,
      strokeWidth: "1.3",
      opacity: "0.65"
    })));
  } else {
    const pts = [[12, 33], [26, 10], [50, 24], [31, 38]];
    marks = /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("path", {
      d: `M${x + 12} ${y + 33} L${x + 26} ${y + 10} L${x + 50} ${y + 24} L${x + 31} ${y + 38} Z`,
      fill: "none",
      stroke: C.ink,
      strokeWidth: "1.3",
      opacity: "0.6"
    }), pts.map((p, i) => /*#__PURE__*/React.createElement("circle", {
      key: i,
      cx: x + p[0],
      cy: y + p[1],
      r: "3.2",
      fill: i === 1 ? C.ink : g
    })));
  }
  return /*#__PURE__*/React.createElement("g", {
    opacity: op
  }, /*#__PURE__*/React.createElement("rect", {
    x: x,
    y: y,
    width: NX_GW,
    height: NX_GH,
    fill: "none",
    stroke: "#C9CCD4",
    strokeWidth: "1"
  }), marks);
}

// ── the act ─────────────────────────────────────────────────────────────────
function ActNext({
  T,
  t0,
  L
}) {
  // Math.min holds the final frame: past 14 s the last authored state stays up
  // rather than stepping into undrawn time.
  const t = Math.min(T - (t0 || 0), NX_DUR);
  if (t < -0.3) return null;
  const labels = L !== false;
  const appear = tw(t, NX_T.wash, NX_T.wash + 0.3, Easing.linear);
  const table = appear * (1 - tw(t, NX_T.out, NX_T.outEnd, Easing.linear));
  const ruleP = tw(t, NX_T.rule, NX_T.rule + 0.6, MOTION.draw);
  const footP = tw(t, NX_T.foot, NX_T.foot + 0.45, Easing.linear);

  // the close
  const ask = tw(t, NX_T.ask, NX_T.ask + 0.4, Easing.linear);
  const qP = tw(t, NX_T.question, NX_T.question + 0.5, MOTION.enter);
  const qRule = tw(t, NX_T.qrule, NX_T.qrule + 0.45, MOTION.draw);
  const th1 = tw(t, NX_T.thesis1, NX_T.thesis1 + 0.45, MOTION.enter);
  const th2 = tw(t, NX_T.thesis2, NX_T.thesis2 + 0.45, MOTION.enter);
  const cRule = tw(t, NX_T.crule, NX_T.crule + 0.45, MOTION.draw);
  const cr1 = tw(t, NX_T.credits, NX_T.credits + 0.45, Easing.linear);
  const cr2 = tw(t, NX_T.credits2, NX_T.credits2 + 0.45, Easing.linear);
  const dis = tw(t, NX_T.disclose, NX_T.disclose + 0.5, Easing.linear);
  const closing = Math.max(ask, qP, th1);
  const rows = NX_ROWS.map((r, i) => {
    const a = NX_T.row[i];
    return Object.assign({}, r, {
      y: NX_ROWY[i],
      // NB: opacity keys are prefixed `op` on purpose. A key named `body` or
      // `title` here would silently overwrite the row's copy with a number.
      opRule: tw(t, a, a + 0.5, MOTION.draw),
      opHead: tw(t, a + 0.08, a + 0.48, MOTION.enter),
      opBody: tw(t, a + 0.26, a + 0.70, Easing.linear),
      opVs: tw(t, a + 0.48, a + 0.92, MOTION.enter)
    });
  });
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      fontFamily: FONT
    }
  }, /*#__PURE__*/React.createElement(NXWash, {
    op: appear
  }), table > 0.001 && /*#__PURE__*/React.createElement("svg", {
    width: "1920",
    height: "1080",
    style: {
      position: 'absolute',
      inset: 0
    },
    opacity: table
  }, /*#__PURE__*/React.createElement("line", {
    x1: NX_L,
    y1: NX_HEAD_RULE,
    x2: NX_L + (NX_R - NX_L) * ruleP,
    y2: NX_HEAD_RULE,
    stroke: C.ink,
    strokeWidth: "2"
  }), rows.map(r => /*#__PURE__*/React.createElement("g", {
    key: r.key
  }, /*#__PURE__*/React.createElement("line", {
    x1: NX_L,
    y1: r.y,
    x2: NX_L + (NX_R - NX_L) * r.opRule,
    y2: r.y,
    stroke: "#C9CCD4",
    strokeWidth: "1"
  }), /*#__PURE__*/React.createElement("line", {
    x1: NX_VS,
    y1: r.y + 34,
    x2: NX_VS + 30 * r.opVs,
    y2: r.y + 34,
    stroke: C.helix,
    strokeWidth: "2",
    opacity: r.opVs
  }), labels && /*#__PURE__*/React.createElement(NXGlyph, {
    kind: r.glyph,
    x: NX_L,
    y: r.y + 56,
    op: r.opBody * 0.95
  }))), /*#__PURE__*/React.createElement("line", {
    x1: NX_L,
    y1: "898",
    x2: NX_L + (NX_R - NX_L) * footP,
    y2: "898",
    stroke: "#C9CCD4",
    strokeWidth: "1"
  })), table > 0.001 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: table
    }
  }, /*#__PURE__*/React.createElement(NXTxt, {
    x: NX_L,
    y: 188,
    op: tw(t, NX_T.eyebrow, NX_T.eyebrow + 0.4, Easing.linear),
    size: 20,
    weight: 600,
    ls: "0.15em",
    color: "#5A5C6B"
  }, "NEXT STEPS \xB7 PROPOSED, NOT RUN"), /*#__PURE__*/React.createElement(NXTxt, {
    x: NX_L,
    y: 224,
    op: tw(t, NX_T.title, NX_T.title + 0.5, MOTION.enter),
    size: 46,
    weight: 600
  }, "Three experiments, each with its control."), rows.map(r => /*#__PURE__*/React.createElement(React.Fragment, {
    key: r.key
  }, /*#__PURE__*/React.createElement(NXTxt, {
    x: NX_L,
    y: r.y + 20,
    op: r.opHead,
    size: 22,
    weight: 500,
    color: "#8A8B90",
    ls: "0.06em"
  }, r.n), /*#__PURE__*/React.createElement(NXTxt, {
    x: NX_TXT,
    y: r.y + 16,
    op: r.opHead,
    size: 34,
    weight: 600
  }, r.title), /*#__PURE__*/React.createElement(NXTxt, {
    x: NX_TXT,
    y: r.y + 70,
    op: r.opBody,
    size: 22,
    weight: 400,
    lh: "1.45",
    width: 1020,
    wrap: true,
    color: "#3C3E52"
  }, r.body), /*#__PURE__*/React.createElement(NXTxt, {
    x: NX_VS,
    y: r.y + 12,
    op: r.opVs,
    size: 17,
    weight: 600,
    ls: "0.18em",
    color: "#8A8B90"
  }, "VS."), /*#__PURE__*/React.createElement(NXTxt, {
    x: NX_VS,
    y: r.y + 48,
    op: r.opVs,
    size: 21,
    weight: 500,
    lh: "1.4",
    width: 356,
    wrap: true,
    color: C.ink
  }, r.vs))), /*#__PURE__*/React.createElement(NXTxt, {
    x: NX_L,
    y: 936,
    op: footP,
    size: 20,
    weight: 400,
    color: "#6A6C7A"
  }, "None of the three has been run. The right-hand column is the comparison, not a result.")), closing > 0.001 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 268,
      textAlign: 'center',
      opacity: ask,
      font: `600 20px ${FONT}`,
      color: '#5A5C6B',
      letterSpacing: '0.15em'
    }
  }, "WE OPENED ON A QUESTION"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 308,
      textAlign: 'center',
      opacity: qP,
      transform: `translateY(${(1 - qP) * 10}px)`,
      font: `400 44px ${FONT}`,
      color: '#3A3C50'
    }
  }, "Sequence Is All You Need?"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 960 - 130 * qRule,
      top: 394,
      width: 260 * qRule,
      height: 2,
      background: C.crimson,
      opacity: 0.7
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 438,
      textAlign: 'center',
      opacity: th1,
      transform: `translateY(${(1 - th1) * 12}px)`,
      font: `600 64px ${FONT}`,
      color: C.ink
    }
  }, "The expensive arm lost."), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 534,
      textAlign: 'center',
      opacity: th2,
      transform: `translateY(${(1 - th2) * 12}px)`,
      font: `400 40px ${FONT}`,
      color: C.ink
    }
  }, "The threshold we declared first is why we can say so."), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 960 - 210 * cRule,
      top: 648,
      width: 420 * cRule,
      height: 2,
      background: C.ink,
      opacity: 0.55
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 706,
      textAlign: 'center',
      opacity: cr1,
      font: `500 22px ${FONT}`,
      color: '#3C3E52',
      letterSpacing: '0.04em'
    }
  }, "logic binders team \xB7 London AI \xD7 Science, Protein Engineering Track \xB7 3\u20134 October 2026"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 752,
      textAlign: 'center',
      opacity: cr2,
      font: `400 20px ${FONT}`,
      color: '#6A6C7A',
      letterSpacing: '0.04em'
    }
  }, "PDB 2BNQ \xB7 1.70 \xC5 \xB7 Rasmussen et al."), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 794,
      textAlign: 'center',
      opacity: dis,
      font: `400 20px ${FONT}`,
      color: '#8A8B90',
      letterSpacing: '0.04em'
    }
  }, "limitations and test-exposure disclosure in REPORT.md sections 2 and 8")));
}
window.ActNext = ActNext;
;
/* src/animations/submission-film/film-scene.jsx */
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

const FILM_W = 1920,
  FILM_H = 1080;

// Authored running order. `dur` is the act's own length; `at` is derived.
// Act I's length is INTRO_LEN below, which must match the intro cue sum.
// Act I keeps four beats and DROPS the rest outright, rather than giving the
// dropped ones a fraction of a second each.
//
// Every beat has a duration below which its own choreography cannot finish —
// the frame at which its last callout, residue pop or ripple lands. Squeezing
// a beat under that does not speed it up, it cuts it mid-motion; at the
// extreme the beat renders nothing at all. The previous 14s Act I ran eight of
// twelve beats under their minimum (Expansion at 3% of what it needs) and its
// final second was a blank frame.
//
// So: Groove, Anatomy, Candidates and Lock at their true minimums, and the TCR
// docking / wobble / activation / clonal-expansion arc removed. The deck gives
// biology one section titled "Peptide-MHC interface", and a results-first film
// does not need the immunology payoff to set up a ranking metric.
//
// Approach is kept only as a terminal cue: BeatGroove fades itself out across
// it, so it is the act's out-point rather than a beat of its own.
const INTRO_CUES = {
  Groove: 1.3,
  Anatomy: 3.8,
  Candidates: 2.1,
  Lock: 1.0,
  Approach: 0.5
};

// Turn the duration map into absolute starts, the same way the runtime derives
// CUES from OM_SCENES, so the intro beats see the cue object they expect.
function introCueStarts(cues) {
  const A = {};
  let acc = 0;
  Object.keys(cues).forEach(k => {
    A[k] = acc;
    acc += cues[k];
  });
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
const ACTS = [{
  key: 'coldopen',
  dur: 4.5,
  comp: () => window.ColdOpen
}, {
  key: 'intro',
  dur: INTRO_LEN,
  comp: null
},
// rendered inline below
{
  key: 'question',
  dur: 12,
  comp: () => window.ActQuestion
}, {
  key: 'approach',
  dur: 12,
  comp: () => window.ActApproach
}, {
  key: 'verdict',
  dur: 20,
  comp: () => window.ActVerdict
}, {
  key: 'transfer',
  dur: 16,
  comp: () => window.ActTransfer
}, {
  key: 'next',
  dur: 14,
  comp: () => window.ActNext
}];
const AT = (() => {
  const m = {};
  let acc = 0;
  ACTS.forEach(a => {
    m[a.key] = acc;
    acc += a.dur;
  });
  m.__total = acc;
  return m;
})();

// The film's one hard cut: four frames of black immediately before the verdict.
const CUT_AT = AT.verdict,
  CUT_LEN = 4 / 30;

// Act I: the intro beats, shifted onto film time. They are defined in
// pmhc-scene-v2.jsx and take the cue object as `A`.
function ActIntro({
  T,
  t0,
  L
}) {
  const t = T - t0;
  if (t < -0.2 || t > INTRO_LEN + 0.4) return null;
  if (typeof BeatHLA !== 'function') return null;
  const A = INTRO_A;
  // Only the two beats whose cues INTRO_CUES actually defines. The dropped
  // beats are not mounted at all: giving them a near-zero cue would still let
  // their opacity math fire for a frame or two, which is what produced the
  // flashes and the blank tail in the previous cut.
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(BeatHLA, {
    T: t,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatGroove, {
    T: t,
    A: A,
    L: L
  }));
}

// An act renders only inside its own window (plus a little overlap for
// cross-dissolves), so one act's stray frame can never sit under another's.
function ActSlot({
  T,
  act,
  L
}) {
  const t0 = AT[act.key];
  const Comp = act.comp ? act.comp() : null;
  if (typeof Comp !== 'function') return null;
  if (T < t0 - 0.3 || T > t0 + act.dur + 0.3) return null;
  return /*#__PURE__*/React.createElement(Comp, {
    T: T,
    t0: t0,
    L: L
  });
}

// Only the biology act needs a tag from the shell. The headline card carries
// its own masthead and every act from the task beat onward draws its own
// header, which is what the guard below enforces.
const FILM_CHAPTERS = [['coldopen', '', ''], ['intro', '01', 'peptide-MHC interface']];

// Acts III-VI draw their own chapter headers at this same corner, so the shell
// only labels the acts that don't: the intro (whose own ChapterTag lives in
// Piece, which we don't mount) and the question act.
function FilmChapterTag({
  T
}) {
  if (T >= AT.question) return null;
  let cur = null;
  FILM_CHAPTERS.forEach(c => {
    if (T >= AT[c[0]]) cur = c;
  });
  if (!cur || !cur[1]) return null;
  const since = T - AT[cur[0]];
  const op = clamp(since / 0.4, 0, 1) * (1 - clamp((T - AT.transfer - 7) / 0.6, 0, 1));
  if (op <= 0.001) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 80,
      top: 86,
      opacity: op,
      display: 'flex',
      gap: 16,
      alignItems: 'baseline',
      font: `500 22px ${FONT}`,
      color: C.ink,
      letterSpacing: '0.06em',
      textTransform: 'uppercase'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 600
    }
  }, cur[1]), /*#__PURE__*/React.createElement("span", {
    style: {
      width: 36,
      height: 2,
      background: C.ink,
      alignSelf: 'center'
    }
  }), /*#__PURE__*/React.createElement("span", null, cur[2]));
}
function Film({
  labels,
  captions
}) {
  const comp = useComposition();
  // window.__FILM_T__ pins the stage to one authored second, for frame review
  // and still export (index.html sets it from a ?t= query param). Playback is
  // unaffected when it is null.
  const T = window.__FILM_T__ == null ? comp.T : window.__FILM_T__;
  const L = labels !== false;
  const cut = T >= CUT_AT - CUT_LEN && T < CUT_AT ? 1 : 0;
  return /*#__PURE__*/React.createElement("div", {
    "data-screen-label": `t=${Math.floor(T)}s`,
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      background: '#E9EBEF',
      fontFamily: FONT
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement(ActIntro, {
    T: T,
    t0: AT.intro,
    L: L
  }), ACTS.filter(a => a.comp).map(a => /*#__PURE__*/React.createElement(ActSlot, {
    key: a.key,
    T: T,
    act: a,
    L: L
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      pointerEvents: 'none',
      background: 'radial-gradient(ellipse 80% 75% at 50% 48%, rgba(43,45,66,0) 60%, rgba(43,45,66,0.14) 100%)'
    }
  }), /*#__PURE__*/React.createElement(FilmChapterTag, {
    T: T
  }), cut ? /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      background: '#000'
    }
  }) : null);
}
function SubmissionFilm() {
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(CompositionStage, {
    width: FILM_W,
    height: FILM_H,
    scenes: window.OM_SCENES,
    playback: window.OM_PLAYBACK,
    bg: "#E9EBEF"
  }, /*#__PURE__*/React.createElement(Film, {
    labels: t.labels,
    captions: t.captions
  })), window.__FILM_PANEL__ === false ? null : /*#__PURE__*/React.createElement(TweaksPanel, null, /*#__PURE__*/React.createElement(TweakSection, {
    label: "Overlays"
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Scientific labels",
    value: t.labels,
    onChange: v => setTweak('labels', v)
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Captions",
    value: t.captions,
    onChange: v => setTweak('captions', v)
  })));
}
window.FILM_AT = AT;
window.SubmissionFilm = SubmissionFilm;