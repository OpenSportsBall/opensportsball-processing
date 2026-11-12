# Backlog

Open processing and ML work, roughly in priority order within each section.
Phases refer to the main OpenSportsBall repo's `docs/PLAN.md`. When an item
is done, delete it here and mention it in the commit. App, API and
infrastructure work is tracked in the main repo's `docs/BACKLOG.md`.

## Phase 0: feasibility on real footage

- [ ] Carry the `processing/spikes/phase0.py` findings into the
      pipeline. From a 5-minute handheld field clip (2026-10-12) and a
      97.5-minute night game on a fixed mount (2026-10-14, recording
      `78618588`, 21 segments per phone, run locally off the phones):
      - Audio sync locks in every segment of the game (most 4 s windows
        clean). Clock drift is not constant: the per-segment fit reads
        -3.9 ppm in segment 0, -7.8 by segment 4, -8.0 by 5, then -8.2 to
        -9.4 for the rest of the game (the phones warming up?); the
        5-minute field clip read -5.9 and -4.6. Still <= ~2.5 ms per
        5-minute segment, so one offset per segment is enough
        (`footage/rec3/phase0/report.json`).
      - Whiten only 150 Hz-8 kHz (partial PHAT, beta 0.8): AAC leaves
        almost nothing outside that band, and whitening it made a false
        peak at zero lag that the search returned as its answer (desk
        test, 2026-10-12).
      - Re-sync every segment by audio. The offset jumps by up to 390 ms
        at segment switches (the phones lose different amounts), and
        `startedAtDeviceMs` doesn't show it: BLE+metadata was 10 ms off
        for segment 0 and 309 ms off by segment 20. At the start BLE is
        within about a frame (5 ms at segment 0 of the field clip; 22 and
        31 ms on two short desk tests, unverified); on the game its error
        grows with every switch (-59 ms by segment 5, -119 by 13, -309
        by 20), so it only seeds the audio search.
      - Stitch: OpenCV's panorama stitcher (estimated focal lengths and
        rotations, graph-cut seam, multi-band blend). One calibration from
        segment 0 held for the whole mounted game: overlap points moved
        ~3 px median at 960 px wide in every segment, the same as the
        measurement noise within one segment. The handheld clip drifted
        (45-57 deg apart), so the pipeline needs a rigid mount or a
        periodic re-estimate. Single-pair estimates of the angle are
        noisy at night (78.8-86.8 deg); compare homographies instead.
        The stitcher's own per-segment re-estimate failed in 9 of 21
        night segments (calibration used 5 of 8 pairs), so "every
        segment" is the homography check, whose overlap points moved
        1-4 px median and up to 6-28 px at worst (near-field rows; the
        maxima unverified).
      - A homography fitted on field points did worse than the stitcher:
        turf texture gives mostly false matches.
      - 97.5 min of 4K on battery: thermal nominal throughout; battery
        100 to 74% (Pixel 11) and 55% (11 Pro); ~24 GB per phone.
      - Second night game (rec4, 2026-10-21, 115 min, 25 segments per
        phone; Pixel 11 Main, Pixel 11 Pro Secondary mounted upside
        down): audio sync locks in every segment, drift -5.6 ppm. The
        offset climbs at every switch, 69 ms (seg 0) to 2.93 s (seg 24),
        ~120 ms per switch; BLE+metadata is 22 ms off at seg 0 and 776 ms
        by seg 24. Stitch 79.6 deg apart on 3 of 8 pairs; per-segment
        estimates 76.6-87.9 deg in 11 of 25 segments plus one degenerate
        (169 deg). Refined on 394 matches: 11.6 -> 5.0 px median at 4K.
      - First iPhone + Pixel game (rec5, 2026-10-28; Pixel 11 Main, iPhone
        14 Pro Secondary; 2026-10-29): the Pixel cuts segments at ~285 s,
        the iPhone at 300 s, so Main's segment k and Secondary's k drift
        15 s apart per segment. Index pairing froze the iPhone half for the
        first 30-135 s of each segment (~23 of 92 min). The start stamps
        were 1.8 s off at the start and 3.9 s by the end (1737 + 107.5 ms
        per segment, residuals <= 17 ms), past sync's +-2 s search, so
        sync failed (sharpness 0.7-1.2 from segment 3). Now phase0.py
        pairs each Main segment with the file that overlaps it most,
        searches around the previous segment's stamp error (+-8 s for the
        first), lists a second overlapping file as secondaryPieces that
        game_stream switches to mid-segment, and counts a sync as
        confident when it is sharp (>= 3) or most 4 s windows agree. All
        20 pairs lock, leftover stamp error <= 42 ms; drift +14 ppm. Raw
        rec5 matches the hand-built rec5-game (offsets within 0.02 ms,
        frames across a switch identical); rec2-rec4 pair as before
        (7cfd1f2).
      - A Secondary that stopped early (the game of 2026-11-11: Pixel 11
        Main 99.5 min in 21 segments, iPhone Secondary 44.9 min in 9
        files; 2026-11-12). phase0.py paired every later Main segment
        with Secondary's last file and died asking it for a frame past
        its end (the mount check: `IndexError` in `frames_at(...)[0]`),
        so `prepare` failed at the sync; past that, game_stream repeated
        Secondary's last frame for the rest of the game, which the
        detectors and the render would have taken as live. Now a Main
        segment with under 10 s of Secondary (or under 30 s that doesn't
        lock) is placed from the stamps and not synced, one with none is
        marked `noSecondary`, the report gives `secondaryCoverage` (the
        stretches of the game Secondary has; gaps up to 2 s count as
        covered, as readers hold a frame across them), and the stitch,
        the camera refinement, the exposure match and the engine's pitch
        and kit sample use segments that have both phones. Where
        Secondary has nothing, game_stream gives blank frames, ballwin
        and roles search Main alone, and follow.py renders Main alone:
        the seam goes to Secondary's far side, blobs count only where
        Main sees, the background is relearnt for 2 s when half the
        panorama changes, and the view stays inside Main's frame
        (corners included). Play on Secondary's side is then not in the
        video: nothing filmed it. Tried on the 2026-11-04 game (Pixel +
        iPhone) with Secondary cut to its first two files: sync,
        cameras, and two 60 s chunks across the end of its footage
        render with no frozen or black half. Not yet run on a whole game
        with a gap, and the app doesn't say that part of the game is
        Main alone. A phones' footage that never overlaps fails as
        `no_overlap`.
      - The stitcher failed on rec5: 0/8 pairs over the game, 1/8 on
        segments 7, 12 and 17 (focal 415/449 px, 72.8 deg apart), with
        overlap features in 3 of 20 segments. The phones overlap on a
        narrow strip of turf and the iPhone half is out of focus. The
        cameras were set by hand (2026-10-29): Main from the manifest's
        FOV and gravity sensor, the angle between the phones from the
        touchline and halfway paint (~81 deg), then rotation, k1 and the
        7 cm lens offset fitted jointly on the overlap; agreement in the
        overlap went from 0.25 / 0.68 / 0.86 to 0.90 / 0.90 / 0.85 near /
        middle / far (unverified: the scripts were not kept), and a
        0.75 deg step at the far goal went away. A game whose views barely
        overlap needs a wider overlap at setup or a calibration from the
        pitch.
      - rec7 (2026-11-04, Pixel 11 + iPhone 14 Pro, 99.95 min): sync locks
        but sharp matches are rarer (mostly 1.5-3 against 3-9 on rec6);
        stamp error ~-2.2 s by the end. Stitch ok on 3/8 pairs, 79.2 deg
        apart, the angle 77.7-87.1 deg over 14/22 segments. phase0.py had
        calibrated from the last settings lock of the whole game (rec7
        re-locked at ~52 and ~75 min), so in effect from all of segment 0,
        warm-up included; rec6 the same. It now takes the last lock within
        segment 0 (rec7 249.5 / 254.9 s, rec6 37.4 s; 1a12e90). rec6's and
        rec7's cameras were made the old way and not rebuilt (that would
        void their pitch fits and taps); it may explain part of rec7's
        weak stitch.
      - Line fit (2026-11-05, rec7, measured only; nothing re-rendered):
        the paint didn't line up at the seam (centre circle split up to
        ~10 px at 4K, worst toward the phones' frame edges). refine() fits
        SIFT matches, mostly sky, trees and false turf matches, never the
        paint. `follow.refine_lines` now fits paint centrelines (median
        background per phone, top-hat + skeleton, pieces >= 150 px) in
        both directions by distance transform, with both lenses, a lens
        offset Camera.t (in camera heights; parallax on a level ground,
        so no pitch map is needed) and a Secondary rotation per sampled
        segment. Held out (segments 6 and 15, fitted on 0/4/8/13/17):
        1.80 -> 0.76 and 2.57 -> 0.82 px median at 4K. Offset alone on
        segment 6: 0.79 -> 0.64 px, 6.6 cm across (the far rows gain
        most, 1.16 -> 0.45). Fitting the ground's tilt too didn't help
        (0.67). Fit on segment 1 vs 6: each 0.6 px on its own and
        ~5.8 px on the other, Secondary rolled ~0.4 deg between them
        (the mount moved early in the game, after the calibration
        window). The kept rotation is the one most segments agree with;
        the others are listed in cameras.json "lineFit.mountMoved".
        Adds ~2-3 min to building cameras.json.
- [ ] Line detection and pitch mapping without taps. Every field since
      rec3 has needed taps; the joint fit (`jointfit.py`, below) fixed the
      lenses but not this. Untapped joint fits, priors from the lens table
      with the recording left out (2026-11-09): rec7 3.1 / 9.5 px tap
      error, rec6 6.4 / 16.0 (about its tapped fit), rec4cal 35.3 / 78.0
      (length runs to 117 m against ~105; 4.1 px with a 1% different
      prior, so unstable), rec5-game fails. And they started from the
      tapped pitch.json: a real untapped game starts worse. Where it fails:
      - Paint in the phones' own frames: `pitch.line_mask`'s scales were
        tuned on the 0.03 deg panorama. rec5-game's far yellow lines go
        undetected; rec6's white mask is so wide that 97% of model points
        sit "on paint", so the paint term says nothing there.
      - The far ends: the end boxes and goal lines are thin and few pixels
        tall, and the fit locks the ends onto the wrong paint (length runs
        away). Junction detection found one box of four on rec4.
      - The start: `best_start` searches camera heights; nothing measures
        the centre circle, halfway line or touchlines to seed the pose.
      Ideas: paint detection per phone at the scale the line should have
      there (its width is known from the current fit); fit the ends
      separately once the middle holds (pitch.fit_end_pose does this in the
      panorama); goals as end anchors (validated on rec3); a learned
      keypoint detector trained on the tapped recordings' `taps_src.json`
      (source pixels, so the labels survive stitch changes).
      Measure: tap error and the rec4 traced benchmark (`jointfit.py
      score` / `bench`) for untapped fits started from `best_start`, not
      from a tapped pitch.json; done when every tapped recording's
      untapped fit is within ~2x its tapped fit.
- [ ] Per-segment Secondary rotation in rendering: refine_lines already
      fits one per sampled segment, but every caller of setup_cameras
      (and game_stream across segment switches) uses one camera pair.
      rec7 segments 0-1 are ~5 px off with the game's rotation.
      - iPhone focus (2026-11-03): rec5's iPhone half was soft throughout
        (focus forced to lensPosition 1.0, past infinity on the Pro
        ultra-wide). With the app's settle-then-sweep lock, a tester's
        40 s iPhone 14 Pro clip locked at lens 0.85 (autofocus had settled
        at 0.851) and the locked stretch is as sharp as the best autofocus
        frames, corners included; the lock came only at 34.6 s of 40.
        Unverified (scratch measurements); not yet seen on a game.
- [ ] Pixel 11 Pro reports a 108.8 deg horizontal field of view but its
      recorded frame is ~104 deg (stitcher's focal estimate); the Pixel
      11's 103.6 deg matches. Check `CameraSelection.horizontalFovDeg`
      (crop or zoom ratio). Processing doesn't depend on it today.
- [ ] A lens model per phone model, so the stitch can't drift in scale.
      The stitcher only sees the overlap, so it can't separate both
      focals from the angle between the phones: the same Pixel 11 came
      out at f 733.6 px @ 1920 on rec6 and 763.5 on rec7 (2026-11-04),
      and the 4% bent rec7's near touchline ~1 deg (pitch.py's w3 now
      absorbs it in the taps refit). Holding Main's lens at rec6's value
      and refitting only Secondary straightened the near touchline but
      broke the far end and centre (overlap error 5.0 -> 8.4 px at 4K):
      both lenses need to be known, not one. Earlier evidence (rec4,
      2026-10-22): refitting k1/k2 so the painted near touchline and
      halfway line come out straight works with focal free (off-plane
      16-20 -> 2-8 px at 4K, stitch 5.1 px) but with focal held the
      stitch degrades to 17.3 px. Factory intrinsics (`dumpsys
      media.camera`, unverified): Pixel 11 752.8 px @ 1920, matching the
      stitcher; Pixel 11 Pro 670.1 @ 1920, but the stitch wants ~726
      (21.3 px transfer at 670, 7.4 at 726) and the touchline/halfway
      angle on the ground is 90 deg at 726-730, 79.5 at 670. rec4cal
      uses Main's factory focal, 726 px for the Pro, no distortion,
      levelled to Main's gravity sensor (the fixture README). Plan:
      - Prior first: `follow.refine` pulls each focal toward the
        manifest's `video.horizontalFovDeg` (f = 960 / tan(hfov / 2) @
        1920: Pixel 11 755.3, iPhone 14 Pro 720.8) instead of fixing it;
        the Pixel 11 Pro's reported FOV is known to be wrong (above), so
        a table per (model, lens, resolution) overrides it.
      - Then measure the table: focal, k1, k2 per phone model from a game
        with tapped keypoints (the pitch is metric, so it pins the scale
        the overlap can't) or a checkerboard.
      - Check: rec7's near touchline straight with w3 = 0; the
        night-2026-10-21 marked-field benchmark (pitchbench.py) and the
        play bench no worse. Rebuilding cameras.json changes every
        panorama angle: existing pitch fits and taps.json need redoing
        (taps can be carried across through the phone pixels).
      - Joint fit (2026-11-09, `processing/spikes/jointfit.py`): taps
        carried into each phone's 4K pixels (`pitch/taps_src.json`), then
        one least squares over both lenses (focal, k1, k2), Secondary's
        rotation, the lens offset (metres) and the pitch pose, on each
        phone's own median background (one segment: rec7 11, rec4cal 12,
        rec6 2, rec5-game 10) and its paint, with the taps. No warp terms.
        Priors: focal from the lens table or the manifest FOV (10% with
        taps), k1/k2 sd 0.01, lens offset sd 3 cm. ~15 s per fit once the
        backgrounds are cached. Tap error at 4K, median / worst (current
        -> joint): rec7 6.9 / 20.6 -> 1.8 / 7.6, rec4cal 2.5 / 35.4 ->
        1.2 / 3.4, rec6 4.9 / 16.5 -> 2.7 / 7.0, rec5-game 6.5 / 49.6 ->
        2.9 / 45.2. rec4 traced benchmark at 4K (`jointfit.py bench`):
        all markings 4.5 -> 3.0 median, centre circle 22.8 -> 1.5, boxes
        ~2.5 -> <1, near touchline 7.0 -> 9.2. rec4cal's camera comes out
        at 3.0 m up against the measured 2.6 (the field drops away, see
        FINDINGS). Lens table (`lenses.json`, fitted with taps): Pixel 11
        747.5 +- 3.4 px @ 1920 (n=4, factory 752.8), Pixel 11 Pro 740.9
        +- 5.2 (n=2), iPhone 14 Pro 718.5 +- 13.1 (n=2; rec5-game's
        iPhone half is out of focus). Without taps, priors from the
        table with the recording left out: rec7 3.1 / 9.5 px, rec6 6.4 /
        16.0, rec4cal 35.3 / 78.0 (length runs to 117 m; 4.1 px with a 1%
        different prior, so unstable), rec5-game fails. The lens table
        doesn't make untapped games reliable: the far ends' paint does.
        Next:
        - Tried in the follow cam (2026-11-09): `jointfit.py mirror`
          builds footage/rec7-joint and footage/rec4cal-joint (same video,
          the joint cameras and pitch; must-see boxes, goals and taps
          carried through the phones' pixels, boxes moved 0.07-0.36 deg;
          pass A redone), `jointfit.py convert-detect` carries a clip's
          ball candidates and roles across (not re-detected). Play bench,
          `goalkick,goalframe,unsure,ball,goalin,ballhold,roles,kickoff,
          peoplesurge` with ballwin's candidates: rec4cal 9/15 -> 12/15
          (mean coverage 68 -> 84%), rec7 4/5 -> 4/5 (85 -> 88%); median
          width 58 -> 59 and 52 -> 53 deg. Variant A alone: rec4cal 10/15
          both (different moments), rec7 0/5 both. rec4cal's fitted goals
          come out 7.19 / 7.57 m wide against 7.67 / 7.78 (full size is
          7.32). Seam paint (`jointfit.py stitch`, rec7, px at 4K, current
          -> joint): seg 1 3.52 -> 5.18 (the mount moved), 6 1.81 -> 1.98,
          11 2.77 -> 2.53, 15 2.47 -> 2.15: the joint fit doesn't fix the
          seam; refine_lines does that. A 20 s render of each looked
          alike, no seam faults. Not adopted yet: 20 moments, ball
          candidates found with the old geometry.
        - Checked and corrected (2026-11-10). The numbers above are for
          the taps each fit was given. `jointfit.py <rec> holdout` deals
          the taps into five groups and scores each by a fit that
          wasn't given it (from `--start <rec>/pre_joint` on the adopted
          recordings). Left-out taps at 4K, median / worst, old stitch +
          warped pitch -> joint (fitted with all in brackets): rec7 6.9 /
          20.6 -> 3.8 / 7.5 (1.6 / 7.2), rec4cal 2.5 / 35.4 -> 2.2 / 19.6
          (1.4 / 4.9; the left corner flag 19.6), rec6 4.9 / 16.5 -> 4.8 /
          9.6 (3.2 / 9.8), rec5-game 6.5 / 49.6 -> 6.1 / 29.7 (3.0 / 17.1;
          11 taps, length 95-106 m across folds). Median halved on rec7,
          unchanged on the rest; the worst tap 1.7-2.7x closer on all.
          The fit itself: `fit()` differentiated one-sidedly a paint term
          that is sampled from an image, and the solver stopped on xtol
          far out. Synthetic rec7 (paint drawn through the adopted
          cameras, exact taps, start at the pre-joint cameras): one run
          ended at cost 7.5 against 0.77 with central differences, 15 cm
          high and 0.9 m long with the taps at 0.9 px; repeating the last
          stage didn't help (7.53 -> 7.53), re-running all four stages
          did slowly (7.5, 6.9, 6.7, 4.4). Now `jac="3-point"`, and the
          last stage repeats until a pass gains under 0.1% and moves no
          parameter 0.05 of its scale (report.json "fit": passes, cost,
          converged; stderr says when it didn't). A second run from its
          own result, pitch length: rec7 101.98 -> 102.00, rec4cal 105.29
          -> 105.35, rec5-game 103.56 -> 103.58, rec6 104.73 -> 105.07
          (the paint term is flat there). The loss: soft-L1 covered the
          priors too, so one 10 sd out cost 2.9, not 50; it now covers the
          paint and taps only. Left-out taps, softened -> Gaussian
          priors: rec7 3.9 -> 3.8 px, rec4cal 3.4 -> 2.2. The lens offset
          prior stays zero-centred at 3 cm a side (the lenses are ~7 cm
          apart, the direction depends on the mount); with it Gaussian
          the offsets come out under 2 cm (rec6's was 4-6 cm).
          New fits against the 2026-11-09 ones: rec7 f 751.7 / 727.7 -> 750.6 / 725.8, Secondary k1
          -0.0103 -> -0.0010, pitch 101.85 x 59.81 -> 101.98 x 59.87 m,
          camera 6.47 m both. rec4cal f 746.3 / 744.6 -> 749.6 / 740.1,
          Secondary k1 -0.0223 -> +0.0012, pitch 105.22 x 61.95 -> 105.29
          x 62.02, camera 2.99 -> 3.07 m (measured 2.6: further off),
          fitted taps 1.2 / 3.4 -> 1.4 / 4.9 px, halfway far end 0.84 ->
          1.75 m on the ground. rec5-game pitch 88.7 x 59.4 -> 103.6 x
          64.9 m, worst tap 45.2 -> 17.1 px. rec6 104.0 x 68.7 -> 104.7 x
          69.2. Under the old objective the new fits cost less than the
          adopted ones (rec4cal 92.0 against 97.3, rec7 136.9 against
          137.9). rec4 traced benchmark, pre-joint -> adopted -> new: all
          4.5 -> 3.0 -> 3.7 median (90th 16.6 -> 16.3 -> 14.4), halfway
          17.1 -> 8.8 -> 3.6, circle 22.8 -> 1.5 -> 1.0, far touchline 2.5
          -> 2.5 -> 2.0, near touchline 7.0 -> 9.2 -> 9.2, goal lines 0.7
          -> 0.7 -> 0.9, boxes about the same.
          Lens table rebuilt from the new fits: Pixel 11
          747.5 +- 3.4 -> 748.6 +- 1.9 (n=4), Pixel 11 Pro 740.9 +- 5.2 ->
          741.6 +- 2.1 with k1/k2 scatter 0.0155 -> 0.0036 (rec4cal's
          -0.022 was the stalled fit), iPhone 14 Pro 718.5 +- 13.1 ->
          718.7 +- 10.0. f, k1 and k2 still trade off within a fit; half
          the horizontal field of view is what they agree on (per lens
          +- 0.1-0.3 deg across recordings) and `lenses` now records it
          (halfFovDeg) with the k1/k2 scatter (distSd), which the prior
          uses as its sd where it is over 0.01.
          Re-adopted on rec4cal and rec7 and lenses.json rebuilt (later
          on 2026-11-10). The four tapped recordings refitted into
          <rec>/joint/ (rec4cal and rec7 from pre_joint, rec6 on segment
          2; the 2026-11-09 fits kept in <rec>/joint_2026-11-09/), then
          mirror, convert-detect, convert-passa and `adopt <rec> <mirror>
          joint_2026-11-09_files` (the replaced files; pre_joint/ is
          still the stitch). Most of both recordings' video is off this
          disk (rec7 segments 0, 6, 11; rec4cal 2-6), so the panorama was
          drawn from the fit's cached per-phone backgrounds (rec7 segment
          11, rec4cal 12) and pass A carried across like the candidates
          (`convert-passa`: 99.6% of rec7's blobs, 97.5% of rec4cal's;
          the rest fall outside both 4K frames and keep their old
          angles), not redone. The move from the 2026-11-09 fit is small:
          angles 0.09 deg median, 0.14 worst on rec7 (0.10 / 0.32 on
          rec4cal), the same image point on the ground 4 cm median, 0.7 m
          worst (0.20 / 2.7 m, the far corners). Play bench with the
          carried candidates, `goalkick,goalframe,unsure,ball,goalin,
          ballhold,roles,kickoff,peoplesurge`, before -> after: rec4cal
          13/15 both (mean coverage 91 -> 92%), rec7 4/5 both (88 ->
          89%). Fitted taps: rec7 1.8 / 7.6 -> 1.6 / 7.2 px, rec4cal 1.2 /
          3.4 -> 1.4 / 4.9. Not rerun: ballwin on the new geometry (the
          candidates are carried), the rec5 no-stitch path with the new
          lens table (rough cameras take their focal from it), fullgame.
          The shared dataset and the S3 backup still hold the 2026-11-09
          pitch maps, cameras and boxes until `sharedata.py upload
          --annotations-only` and `footagestore.py backup` are run.
          Untapped fits not retried. `adopt` is still not atomic and
          `lenses` still drops recordings it isn't given.
        - Adopted on rec4cal and rec7 (2026-11-09, `jointfit.py adopt`):
          the joint cameras.json, pitch.json, goals, taps (rebuilt in the
          new panorama's degrees), panorama, must-see boxes and pass A
          caches; every ball candidate and role file carried into the new
          angles. The replaced files are in <rec>/pre_joint/ (README; copy
          them back to undo). fullgame/, phases/ and older renders stay in
          the old geometry. footage/bench/rec4cal is unchanged (the
          standard bench baselines). ballwin rerun on the bench clips with
          today's code and each clip's original checkpoint, on both
          geometries (the old one in footage/rec4cal-old, rec7-old, built
          from pre_joint/; rec7 with OSB_JUDGE=judge_v4, as its 2026-11-05
          files were made): best variant rec4cal 11/15 -> 13/15 (mean
          coverage 73 -> 91%), rec7 4/5 both (85 -> 88%, aim at the ball
          64% both); 15/20 -> 17/20. Candidates per clip about the same on
          both geometries (rec7 5303 / 5336), so the geometry doesn't change
          what ballwin finds. (A first rec7 rerun with the standard judge
          gave ~2200 candidates and 0/5 on both geometries; d9634ff blamed
          ballwin's scan plan or Core ML for that, wrongly. ballwin now
          writes the judge into each ball file.)
        - Next: per-segment Secondary rotation (rec7 segments 0-1).
        - Paint in the phones' frames: line_mask's scales were tuned on
          the 0.03 deg panorama; rec5-game's far yellow lines and rec6's
          over-wide white mask (97% of model points "on paint") need a
          look before untapped fits can work.
        - More recordings with taps per phone model (iPhone 14 Pro has two,
          one of them soft).
- [ ] Players-only follow: `processing/spikes/follow.py` renders a 1080p
      virtual camera straight from both 4K sources. Cameras refined at
      1920 px (median error 11 to 4 px at 4K); each pixel comes from one
      phone along a per-frame seam that avoids players, since the phones
      never capture the same instant and blending doubled moving players;
      two-band blend hides the exposure step. Warping and blending run on
      the GPU (`gpu.py`, PyTorch: MPS locally, CUDA in the cloud), NV12 from
      hardware decode to hardware encode; 20 s of video in 64 s on an
      M-series Mac. Crosses segment boundaries since 2026-10-16
      (game_stream, decoder pre-roll).
- [ ] Follow cam aim, judged by eye on 2 minutes of the night game
      (segment 9, 60-180 s; 2026-10-15). Best so far: "A",
      `follow.py ... play goalkick,goalframe,unsure` (hotspot + goal kicks
      + goal in frame + zoom out when unsure): the only version that
      catches every key moment in the clip, but zoomed out at times
      (median 55 deg). Runner-up: `play goalframe`. All tuned on this one
      clip; check another stretch of the game. What each step fixed:
      - Speed-weighted players beat the median of all players (caught a
        quick counter; in play.py's test, ball in a 60 deg view 100% vs
        66%), but an average of 3-6 runners jumps between groups. The
        point the runners converge on did best at 50 deg on a 20 s test
        (78% vs 63-65%) but jumps more (aim error 12 vs 9.7 deg),
        scored against the early tracker only (509c43a); not wired in.
      - Hotspot of running (pooled +-0.5 s, weighted by speed, sticky:
        another spot must be 50% stronger for 0.7 s): followed the attack
        to the left goal; aim jumps > 40 deg/s dropped 435 -> 73 frames.
        Needed linear (not squared) speed weights and 0.5 s position
        smoothing: far-side jitter read as running.
      - Aim only from people inside the lines; spots by a touchline
        occupied >= 10% of the time (bench, subs) are ignored.
      - Lull hold: <= 1 runner for 1 s keeps the aim from just before.
        The background model learns over ~1 min instead of ~10 s, so
        still keepers stay visible longer (only partly: see roles).
      - goalframe: >= 4 players within 20 m of a detected goal while the
        hotspot is elsewhere: frame both and the goal mouth (<= 75 deg).
        Only worked once goals were detected (pitch.py goals): with the
        model's goal lines it framed the wrong place at the right end.
      - goalkick: a lone near-still player in/near a penalty area soon
        after play was at that end: frame taker and group, held as a
        state until 3 players run for 0.5 s (max 12 s). A keeper 50 m away
        is tracked only on and off, so the pattern is judged over +-1 s.
      - unsure: <= 1 runner near the hotspot keeps the view >= 55 deg,
        <= 2 >= 42 deg (a quick clearance leaves the players behind).
      - lead (aim 1 s ahead): no clear gain in this clip.
      - Hotspot weights, on `play goalframe` (2026-10-15): `nowalk`
        (players under 2 m/s weigh nothing) frames tighter (median 30 vs
        36 deg), goalframe fires ~20% less, and the aim moves elsewhere
        for a few seconds (0:29, 0:42, 0:57); `fastboost` (3x the slope
        above 4 m/s) changes almost nothing: fast runners already
        dominated and the switch rule damps them. Neither adopted.
      - ball (`ballclip.py` candidates from the fine-tuned detector at
        10 fps, `ball.py`'s chain, aim at the ball where tracked >= 0.5 s,
        goalframe and lull hold off there): on `play goalframe` it followed
        the attack to the left goal at :44 with the goal mouth in view,
        but the ball was tracked only 39% of the clip (none after 1:34),
        so it missed the 1:55 goal kick. Next: A + ball (the ball replaces
        unsure's widening where it's known), motion candidates and a
        better tracker to fill the gaps.
      Not solved without the ball: a long clearance until players react.
- [ ] Follow-cam evaluation set (2026-10-16): `playeval.py` builds a
      full-resolution panorama at chosen moments; the page in `evalpage/`
      records a reviewed "must-see" box per moment (panorama degrees);
      `playbench.py` plans every variant from cached player detections
      (seconds per clip) and scores it: moments captured (>= 98% of the
      box in view) and view width. 13 reviewed moments on two 2-minute
      clips (segment 9 from 60 s, segment 3 from 225 s, crossing into 4).
      Note the earlier renders' paths were ~0.7 s off the video (decoder
      frame repeat after a seek, fixed in frame_stream), so judgements of
      those renders don't line up with these scores. Results:
      - A (goalkick,goalframe,unsure): 4/13, median width 54 deg. Misses
        are aim, not width: the hotspot stays on the chasing pack while
        a lone runner has the ball, or the shot happens at the goal ahead
        of the pack; a throw-in taker at the bench is filtered out.
      - Hotspot tweaks without the ball (weak, nowalk, allnear, flow,
        lone, goalin, gk2, bigger bandwidth, no lull): none beat 4/13.
      - A + ball (fine-tuned detector + chain, 10 fps): 8/13 at 54 deg.
        A + ball + goalin (goal mouth kept in frame when the hotspot is
        within 30 m of a goal): 9/13, mean coverage 85%, same width:
        the best principled result. + nowalk (2 m/s) reaches 10/13 but
        wider (68 deg) and mostly by accident (a wide goal-kick frame).
      - Still missed: a goal kick where the ball sits in the keeper's
        hands (seg 9, 70 s: tracker loses it, camera lags), a throw-in
        taken from the bench spot by the camera (seg 3, 105 s: the taker
        is 0.1 m outside the line in a loiter cell), and the clip's first
        seconds (seg 3, 2.5 s: a lull at frame 0; not a full-game issue).
      Ball tracking (2026-10-16): the fine-tuned detector finds the match
      ball in 65-72% of clip frames (conf >= 0.1) but ball.py's chain
      keeps 23-39%: tracks fragment (median 1 hit; a 0.3 s gap ends one).
      Tolerating longer gaps raises coverage to 75% but captures fall
      (9 -> 4-6/13): the extra frames follow false tracks. `balljudge.py`,
      a 48 px crop classifier trained on the labels (4 min on mps, 1.5 ms
      per crop), ranks candidates about as well as possible (held-back
      moments: match ball top-1 in 62% with the stock detector's
      suggestions, near the 77% candidate ceiling) but filtering the
      tracker's candidates with it doesn't lift the captures (8/13).
      Motion candidates alone cap at 55%: half the misses are the ball
      merged into a player's blob; the detector finds 92% of those. So
      the production shape is detector in search windows around the
      track (+ periodic full scans), judge to clean candidates, tracker
      over confidence x judge. To tune the tracker itself we need ball
      ground truth along a clip (labels every ~2 s), not 13 play boxes.
      Clip ball labels (2026-10-16, `labelset.py clip`: a both-phone
      moment every 2 s along each clip, 120 moments, reviewed): with
      `balleval.py` the tracker is measured directly. Found at labelled
      moments, seg 9 / seg 3: as it was 32% / 26% (wrong 9 / 12); with
      the static filter at 30 s (at 4 s the ball placed for a goal kick
      was removed as a spare: 12 -> 18/37 on seg 9 alone, 6027fc8), a
      2-hit minimum, judge >= 0.3 and a 0.6 s gap: 54% / 62% (wrong 1 /
      8; with a looser tolerance 57% / 66%, wrong 0 / 4). Now the
      defaults. Label provenance: both clip sets were pre-marked by
      Claude from the detector and judge, then reviewed (on seg 9 the
      review moved 17 of 31 ball marks by 20-110 px, removed 2 and added
      13; unverified). 20 of seg 9's 118 frame labels and 33 of seg 3's
      102 still carry `draftBy: claude` (unedited drafts, mostly where
      the ball was too small to be sure), so the clip scores lean on the
      detector's own view in those frames. Still unmeasured on the
      labels: ball.py's body filter (BODY_FRAC 0.8, candidates in the top
      80% of a person's box dropped), though half of motion's misses are
      a ball against a player. Of the remaining misses about
      half have no detector candidate near the ball at all, the rest are
      balls on the near touchline just outside the pitch margin, in a
      player's hands (the judge scores those ~0), or in tracks the chain
      didn't pick. Also fixed: ballclip.py only read the first segment
      file, so seg 3's candidates stopped at 1:00; with the full clip
      A + ball + goalin captures 10/13 play boxes with the old tracker
      settings (mean coverage 93%), and 8/13 (90%) with the label-tuned
      ones (the seg 3 throw-in at 105 s 100 -> 37%: the judge rejects a
      ball in the thrower's hands; unverified). The trade was accepted
      because the labels measure the tracker directly and 13 boxes
      don't; round two below recovered the boxes.
      Search-window detector (`ballwin.py`, 2026-10-16): an online track
      on the pitch, a native 640 px window plus a 2x zoom window for
      small balls in each phone, full scans every 2 s or when lost. Same
      recall as full scans within a few points (seg 9 49% vs 54%, seg 3
      58% vs 58% with a 3 m pitch margin, now the default) at ~6-8x
      fewer tiles: 6.5 min for 2 min of 4K on the Mac instead of 30.
      Ceiling check: a usable candidate from either detector exists at
      only 24/37 and 32/53 labelled moments (~62%); the tracker reaches
      49-58%. The gap is detection: 27 of 34 no-candidate moments are
      balls under 8 px, 43-100+ m away; ~8 are judge false negatives
      (ball in hands, at the keeper, against a fence). Retraining the
      judge with the clip labels (train on one clip + the random set,
      test on the other) raises its crop AP on the held-out clip to
      0.92-0.96 but leaves the tracker's found-rate unchanged (seg 3
      55% vs 58%, seg 9 46-49% vs 49%): not the lever.
      Ball witnesses from other motion (2026-10-16/17, scored on the
      clip labels; `witness.py`, `flowcands.py`, `facing.py`):
      - Keeper bearings (the keeper stands on the ball-goal line): the
        "keeper" from pass A is a defender 14-17 m out, because a still
        keeper is absorbed into the background model; bearing errors
        11-52 deg median, the point 40-46 m off. Needs a real person
        detector; as a gate it only hurt the tracker.
      - Motion candidates (double difference over three frames, size and
        consistent step, judged): 264 blobs/frame before the judge, 0.6
        after; alone 11-19% found with 16 wrong; in union with the
        window detector 1-3 more moments have a candidate but the
        tracker does worse (43% vs 49%, 34% vs 58%): the extra blobs
        make wrong tracks. Differencing doesn't reach the 5 px far balls
        either; 54 min per clip on the Mac.
      - Net ripple (share of the goal mouth that changed per frame):
        the left goal flags the shot at 0:45-0:50 and the goalmouth
        scramble at 1:30; specific (1 of 23 far moments) but only fires
        when the ball reaches the net (2 of 8 and 3 of 18 near-goal
        moments). The far goal's region is too small at this resolution
        (noise: active at half the far-away moments). Useful as a
        shot/goal event for the follow cam's near goal, not as a ball
        position.
      - Facing direction (COCO keypoints on each tracked player, a ray
        from the shoulders and nose; facing found for ~75% of players):
        single rays within 30 deg of the ball 37-41% of the time (random
        17%), median 38-45 deg off; the best combination (a vote over
        rays) lands a median 28 m from the ball, within 8 m at 5/37 and
        9/53 moments. A weak prior only at this player size.
      None of the four moves the tracker's found-rate; the limit stays
      the detector's reach on tiny far balls.
      Follow cam, round two (2026-10-17, play boxes): the zoom floor
      raised from 30 to 50 deg (a 1080p view narrower than ~55 deg
      upsamples the 4K source: ~26 px/deg at a phone's centre, 65 deg
      for pixel-for-pixel), ballwin.py's candidates as the ball source,
      and "ballhold" (in a lull, stay on the ball seen in the last 2 s,
      with the players around it; the stillness test dropped) take
      "A + ball + goalin" from 8-10/13 to 11/13 captured at a 64 deg
      median width, mean coverage 98%. Misses: the seg 9 goal kick
      (76%: the ball in the keeper's hands, the camera lags) and seg 3's
      first seconds (95%). "netevent" (keep a goal in frame for 2-3 s
      after its net moved) is implemented but costs a box on seg 3 (a
      side-netting hit at 0:50 pulls the view off the dribble at 0:54):
      off by default.
      The view never crosses the canvas's left/right edge: it stops there
      and zooms in instead (EDGE_MARGIN_DEG 2, floor EDGE_MIN_HFOV 40).
      MIN_HFOV, SMOOTH_S, MAX_PAN_DEG_S and BALL_HOLD_S 2-6 were all
      neutral at 11/13 then.
      Ball stage cheaper (2026-10-17; transcript figures, no log kept):
      every other frame (5 fps) finds 16/37 and 30/53 against 18/37 and
      31/53, play boxes 10/13 against 11 (the 1:45 throw-in lost); full
      scans every 4 s (1 s when lost) instead of 2 s: seg 9 15/37 (41%
      vs 49%) at 4.6 tiles per frame. Neither adopted.
      Multi-frame heatmap detectors (2026-10-18, `ballheat.py`, scored
      per labelled frame on both phones: a candidate within 12 px or 1.5
      ball diameters of the label): WASB and TrackNetV2 with their
      authors' football weights (ISSIA-CNR: static 1080p cameras on a
      pitch, trained at 512x288), TrackNetV3 with its badminton weights,
      each run on 512x288 tiles of the 4K frame at native, 2x and 4x
      downscale (64 tiles per frame, ~5 s per frame per model on the
      Mac, unoptimised). The fine-tuned RF-DETR (ballclip.py full scans)
      at the same frames: seg 9 13/48 (27%; 44 of the 48 balls are under
      8 px) at 1.0 candidate per frame, seg 3 35/58 (60%) at 0.9. Raw,
      WASB finds more only with far more clutter (seg 9 23/48 at 70 per
      frame, seg 3 42/58 at 108; at <= 1 per frame it is below RF-DETR).
      With balljudge.py over its blobs (score >= 0.05, judge >= 0.3)
      WASB alone matches RF-DETR, 14/48 at 1.4 per frame and 35/58 at
      1.7, and the two are complementary: their union finds 21/48 (44%)
      and 45/58 (78%). Raw WASB at a low threshold finds nearly every
      ball of 8 px or more (seg 3: 23/25 vs RF-DETR's 15/25). TrackNetV2
      is a little behind WASB (union 44/58), TrackNetV3 far behind with
      10x the clutter. By scale, the far balls come from the 2x and 4x
      tiles, hardly from the native ones.
      Found on the way: labelset.py's images are 0.2-0.9 s later than
      their stored time. phase0.frames_at returns the first frame the
      decoder yields after a seek, up to a keyframe interval late: seg 9
      main +0.38 s, secondary +0.66; seg 3 main +0.86, secondary +0.17
      (ballheat.py measures it per frame by matching the image). Read at
      the nominal time RF-DETR's recall looks like 17% / 38% instead of
      27% / 60%.
      Label times fixed (2026-10-18): the measured offset is stored per
      frame in items.json ("dtS"; `ballheat.py <set> dt` writes it) and
      balleval.py reads the tracker at clipT + dtS of the phone whose
      image was labelled (the two phones of a moment differ; OSB_NOMINAL=1
      gives the old comparison). Same tracker and candidates, nominal ->
      label time, found / wrong / missed: seg 9 win 18/37, 4, 15 ->
      17/37, 1, 19; full scans 18/37, 1, 18 -> 17/37, 0, 20; seg 3 win
      31/53, 8, 14 -> 35/53, 1, 17; full scans 33/53, 8, 12 -> 37/53, 2,
      14. Most of the "wrong" were the clock: with `win` candidates the
      tracker is at 46% / 66% found with 1 wrong on each clip, and the
      rest are gaps (no estimate), not false tracks. labelset.py now
      grabs frames with a 1.5 s preroll (phase0.frame_at: the frame at t
      is identical with a 1 s and a 2 s preroll, and frame_at(t + dtS)
      reproduces the old label images to JPEG noise), so new sets carry
      dtS 0. The random set (labels/) is unaffected: its labels sit on
      its images and nothing reads its times except motion.py, which
      seeks the same way and lands on the same frame. Also fixed: an
      empty OSB_BALL_TAG matched `.both.json` by glob order in balleval.py
      and ballheat.py (the 2026-10-16 full-scan figures predate that file).
      WASB in ballwin.py (2026-10-25, `OSB_WIN_WASB=1`): the clip decoded
      at 30 fps so each 10 fps frame has its neighbours; WASB in fp16 with
      the crop, downscale and normalisation on the GPU (21 ms per tile on
      the Mac against ~80 ms in ballheat.py; same top peaks within a few
      px) on the 2x and 4x tiles in every full scan and one window around
      the prediction (2x below ZOOM_BELOW_PX, else 4x); peaks above 0.05
      (at most 30 per phone) not on an RF-DETR box become candidates,
      conf = the peak, judged inline. Ball stage ~1.5-1.7x (30 s of rec4:
      3.1 -> 5.3 min; WASB itself 30-40% of a run, the rest the 30 fps
      decode). Baselines (`base`) re-run with the same code for rec3
      seg0-0 and seg0-600, whose `win` files predated 2026-10-19; that
      alone moved rec3's play boxes 20 -> 18/26 (6:30 and 7:22 lost).
      Tracker on the clip labels: seg 3 39 -> 43/53 found (wrong 2 -> 1),
      seg 9 18 -> 19/37 (wrong 2 -> 1). Play boxes (peoplesurge default):
      rec3 18 -> 16/26 (gains 7:22; loses 22:36 100 -> 90%, seg 3 2.5 s
      100 -> 90% and 105 s 100 -> 47%), rec4 9 -> 10/15 (gains 18:27 0 -> 100%; 28:04 29 -> 0%), so
      27 -> 26/41. At the lost seg 3 moments the two paths agree within
      ~1 m (WASB has a 1 s gap at 103-104 s, a short bridge at 3 s): the
      change is in the framing, and the bench moves by +-2 on a candidate
      rerun alone. A better ball track, no play-box gain yet: off by
      default: fullgame.py doesn't set OSB_WIN_WASB (exporting it would
      reach ballwin.py, but the output keeps the `.win` name and chunks
      whose `.win.json` exists are skipped, so move those first). Also:
      ballwin.py exits without writing when no frame was decoded (the
      segment files not on disk), instead of an empty candidate file.
      Next: keep WASB candidates out of the raw-candidate rules
      (kick-off, far-ball goal-kick veto, which read conf >= 0.3 of any
      source), and calibrate their conf against RF-DETR's in the path's
      node cost.
      WASB fine-tuned with the background (2026-10-27, `heattrain.py`):
      a median of 25 frames per segment and phone (the empty pitch, 2x
      downscale) as a fourth input frame, its first-layer weights zero
      at the start (so the stock model exactly); trained 3000 steps at
      batch 8 (0.8 s a step on the Mac; batch 16 needs ~22 GB and swaps)
      on 512x288 crops at 1.6-4.5x from 4K, half around a ball, WASB's
      disk target and weighted BCE. Data: the clip sets and the random
      set's frames whose segments were on disk (163 of 408; the S3
      sign-in had expired), 347 frames, 6 GB. Held out per clip (the
      other clip and the random set minus 20 s around it), full scans
      at 2x/4x: seg 3 at ~1.6 peaks a frame 19% -> 62% of in-play balls
      (stock at >= 0.3, tuned at >= 0.5), judged best 40% -> 67%; seg 9
      17% -> 38%, judged 19% -> 38%. Through ballwin.py and the path it
      does not carry over: the tracker found 62/90 with stock WASB and
      61/90 tuned (both with WASB not steering the online track, so
      the RF-DETR candidates are the same in 1135-1197 of 1200 frames).
      Why: of the tuned run's 29 misses, 13 have no candidate within 2
      frames, mostly 50-100+ m out, because WASB looks only in the
      window around RF-DETR's track between full scans; scanning every
      frame (OSB_WASB_EVERY, 32 tiles, +28 min per 2-minute clip) cuts
      those to 10 but the path then skips usable candidates at 12 and
      found falls to 52/90 (wrong 9): with more candidates the linker,
      tuned on RF-DETR's confidences, picks wrong. Letting confident
      tuned peaks pass the judge is worse (seg 9 9/37). Found on the
      way: drop_jumps kept a 0.8 s blip 4 m nearer the players over a
      29 s run of the match ball; now the players decide only when one
      run is >= 10 m (JUMP_PLAY_MARGIN_M) farther from them, else the
      shorter run goes (base and stock-WASB scores unchanged, play
      boxes 27/41 unchanged moment by moment, 20:27 kept). Also:
      ballwin.py's WASB no longer steers the online track by default
      (OSB_WASB_STEER=1 for the old behaviour); OSB_WASB_CKPT loads a
      tuned model, OSB_WASB_MIN (0.3 for a tuned one) its threshold.
      Next: the linker, not the detector. Score each candidate as a
      probability of the ball from (source, conf, judge, size) fitted
      on the labels, use that in the path's node cost, and re-tune the
      path costs with dense candidates (every-frame WASB); then the
      remaining random-set segments (`aws sso login --profile sbv`)
      and pseudo-labels from long confident runs.
      Linker fixes (2026-10-28, `ballpath.py`, `ball.py`): a learned
      P(ball) per candidate has nothing to learn from yet: at the
      labelled moments almost every candidate the path sees is the
      ball (a handful of negatives in 90 moments). The misses instead
      fall in three groups, each fixed in the linker. (1) The match
      ball beyond the 3 m pitch margin: far balls whose depth is
      metres out and balls just over a line, with judged candidates
      thrown away (6 moments); PITCH_MARGIN_M 3 -> 10. (2) A ball at
      rest (goal-kick spot, free kick) the detector loses for just
      over SKIP_MAX: a candidate may now follow one within REST_M (2 m)
      up to REST_SKIP_S (2 s) earlier, bridged. (3) Widening the margin
      let one run stray onto an off-pitch ball at its end; drop_jumps
      then dropped the next 9.6 s run of the match ball as the shorter
      "second ball". Two runs both >= JUMP_KEEP_S (5 s) are now both
      kept, but only when distance from the players cannot tell them
      apart (rec4's spare ball at 20:27, 50 m from play against 23 m,
      is still dropped; keeping all long runs cost that box). Tracker
      found, seg 3 + seg 9: base 57 -> 66/90 (wrong 4 -> 5), stock WASB
      62 -> 70 (2 -> 3), tuned WASB 61 -> 69 (3 -> 3). Play boxes 29/41
      before and after (the same moments; today's base bench reads 29,
      not the 27 of earlier runs). Path costs (NONE_COST, GAP_PENALTY,
      BRIDGE_S, MIN_JUDGE) re-swept on top: +-1-2 each way, within the
      noise of 90 labels, left as they were. WASB's linker gap is
      closed (70 vs 66 base), but it still does not move the play
      boxes; the remaining misses are mostly moments with no candidate
      at all (50-100+ m) and fast balls seen every few frames.
      Far balls and scan rate (2026-10-28/29, `ballwin.py`, scored on
      seg 3 + seg 9's 90 labels). On the labelled frames themselves a
      native full scan finds 38 of the 77 balls under 8 px (49%), a 2x
      pass over the far band 43, and any method, a 2x window right on the
      ball included, 44: close to this detector's single-frame limit.
      2x far band (OSB_FAR_ZOOM=lost|full, off by default: the 320 px
      crops covering the on-pitch pixels where the ball would be under
      ZOOM_BELOW_PX): found 66 -> 67/90 either way, ball stage ~30%
      slower. Scanning more often while lost (OSB_LOST_SCAN_S, found /
      wrong): 0.5 s 66/5, 0.3 s 66/5, 0.2 s 70/3, 0.2 s + far band 70/4.
      0.2 s costs 13-29% more time per clip (rec3's three 10-minute bench
      clips 164.5 -> 192.4 min, +17%). Play boxes on rec3 with fresh
      candidates (the old `.win` files give 20/26): 0.5 s 18/26, 0.2 s
      19/26. The two boxes lost (6:30, 7:22) were tracked densely in
      the fresh candidates and then removed by ball.static_spots: span
      and hit count ignored gaps, so the match ball crossing the same
      spot twice 30 s apart was a "static" ball (190 spots in rec3 0-10
      min at 0.5 s, 73 in the old files). Fixed: a spot is static only
      when a run of sightings with no break over STATIC_GAP_S (10 s)
      meets the span and hit minimums (spots 190 -> 24, 73 -> 22).
      Found / wrong with the fix: old `.win` 66 -> 68/90, 3; 0.5 s
      66 -> 68, 3; 0.2 s 70 -> 71, 3 -> 5. Play boxes: old `.win` 29/41
      unchanged; rec3 0.5 s 18 -> 20/26, 0.2 s 19 -> 20/26 (coverage
      88% -> 90%); rec4cal 0.5 s 9/15, box for box the old files' score,
      0.2 s 10/15 (gains 18:27 0 -> 100%; 28:04 24 -> 14%, 16:37 95 ->
      90%; coverage 68% -> 74%), so 29 -> 30/41. On rec4cal 0.2 s costs
      more: 63.5 -> 87.9 and 69.3 -> 94.2 min per 10-minute clip (+36-
      38%; twice the full scans, the ball is lost more at night).
      0.5 s stays the default for now: 0.2 s buys one box and a label
      or two for 17-38% more ball-stage time, and the far band nothing.
      Orange ball (2026-10-29/30): rec5's match ball is orange, and the
      judge, trained on rec3's white ball, rejected it (5% of frames
      judged ball, path 7% of rec5 0-10 min when rescored with today's
      ball.py; the render itself aimed at the ball 3% of the time). On-
      pitch candidates in 26% of rec5's frames against 52-59% on rec4cal
      (conf >= 0.7: 7% vs 23-29%); 300 confident candidates the judge
      scored < 0.1 were crisp crops of the orange ball. Rescored without
      the judge the path covered 43% of the 10 minutes (unverified): the
      judge was the main block. 98 labels on rec5's Pixel
      (labelset.py ball mode; 71 ball) went into a detector trained on
      rec3 + rec5 (rec5 x3, ballmerge.py; 275 white boxes, 54 orange) and
      a judge on rec3's crops + rec5's (x3, `balljudge.py merge`). rec5
      path 7% -> 34% (old detector + new judge 25%; 29/32 sampled path
      frames on the real ball). White-ball games: rec4cal play boxes
      9/15 for every combination (aim at the ball 39/25% -> 37/22%, new
      judge alone 42/27%); rec3 boxes 20 -> 20/26, balleval 68 -> 67/90
      found, wrong 5 -> 6 (today's base reads 5, not 3; new judge alone
      21/26 but seg9 23 -> 19). The pair is the standard for every game
      since 2026-10-30 (footage/models/). Next: a second orange-ball
      game, and more varied labels so one model gains on both.
      Daylight (2026-10-31/11-01): rec6, the first daylight game, gave
      100 random moments labelled on both phones (196 frames, 83 ball
      boxes). Detector and judge retrained on rec3 + rec5 x3 + rec6 x2
      (balltrain_v3, judge_v3; effective mAP 0.351, judge valid AP
      0.910). Against the rec5 pair: rec6 0-10 min path 30% -> 52%
      (30-31/32 sampled path frames on the real ball; some labelled
      moments fall in those 10 minutes), rec5 39% -> 39%, rec4cal play
      boxes 9 -> 10/15 (aim 37/22% -> 36/19%), rec3 boxes 20 -> 20/26,
      balleval 67 -> 63/90 found, wrong 6 -> 3. The new detector with
      the old judge was worse (rec6 29%, rec3 58/90), so they go
      together. Standard since 2026-11-01 (footage/models/). rec3 seg3
      lost 4 found (46 -> 42/53): watch white-ball games. judge_v3's
      effective mix was rec3 x4, rec5 x9, rec6 x2 (the merge bug below).
      Judge with the expected ball size (2026-11-01): the pitch map's
      expected ball diameter at the candidate as an extra judge input,
      trained with the same seed and weights as without it. Found on the
      way: `balljudge.py merge` read rec5-game's crops.npz, itself a
      merge, so the standard judge_v3 trained on rec3 x4, rec5 x9, rec6
      x2, with duplicates across valid/test (fixed in 95eeac3:
      crops_<rec>.npz). v3 / fixed mix (v3b) / fixed + size (v3d): rec6
      path 52 / 49 / 48%, rec5 39 / 41 / 40%, rec4cal boxes 10/15 each
      (aim 36/19, 30/18, 40/21%), rec3 boxes 20 / 20 / 21/26, found 63 /
      65 / 65 of 90 (wrong 3 / 2 / 4); held-out clip AP 0.915 / 0.915 /
      0.936 (`footage/benchscan/judge_dist.log`). No clear gain: the size
      input was dropped, the merge fix kept, judge_v3 stays the standard.
      Yellow ball (2026-11-05, rec7: white vs yellow kits, yellow match
      ball, night): the first render aimed at the ball in 0-1% of each
      10-minute piece, and 0/5 reviewed play boxes on piece 2 (set
      seg0-1484). The detector finds the yellow ball (conf 0.7-0.86); the
      judge, which had never seen one, scores it 0 (752 of 805 on-pitch
      candidates at 0). 119 frames labelled on both phones (labels_seg6:
      seg 6, 100-220 s; 72 balls) and a judge_v4 trained on rec3 + rec5 x3
      + rec6 x2 + rec7 x4 (valid AP 0.896, held-out clip 0.92). Confident
      candidates passing the judge: 0-5% -> 55-57%. Piece 2 (contains the
      labelled clip, so not independent): aim at the ball 1% -> 64%, play
      boxes 0/5 -> 4/5 (unverified; the miss, 3:21, is a lost track).
      Re-rendered with OSB_JUDGE=judge_v4 from piece 2: aim 40, 28, 43,
      48, 63, 46, 61% on pieces 3-9. Benched against judge_v3 on the
      same saved candidates (2026-11-09): rec3
      play bench even (A+ball 17/26 both; A+ball+hold+nowalk 20 -> 21,
      but play 3 drops 90% -> 0-33% in every ball variant); rec3
      balleval seg3 found 42 -> 41/53, seg9 21 -> 16/37 with wrong
      1 -> 4; rec4cal play bench the same (best 11/15 both); ball path
      share rec5 39 -> 40%, rec6 52 -> 45%; rec7 engine chunks
      1200 / 3000 / 4800: 1 / 0 / 0% -> 47 / 17 / 40%. So it fixes the
      yellow ball but costs on rec3 seg9 and rec6: not the standard yet.
      Each new ball colour has so far needed its own labels.
      Offline path tracker (2026-10-19, `ballpath.py`; the default
      in balleval.py and follow.py, OSB_BALL_CHAIN=1 for the chain): the clip is on disk
      before tracking runs, so instead of ball.py's greedy online chain
      (tracks die after 0.6 s, median one hit) the match ball is one
      minimum-cost path (Viterbi) through the same filtered candidates:
      a node per candidate at -log(conf x judge), a "no ball" node at 2
      per frame, edges to a candidate up to 10 frames back (a skipped
      frame costs the same as "no ball" but needs no gap penalty),
      entering or leaving "no ball" 2. Motion between candidates of one
      phone is angular in that phone's view (linear, 2 deg per frame
      costs 1, steep beyond 8): on the pitch plane a ball in the air
      runs out and back along the camera's ray at 5-14 m per frame, and
      any metric cost rejected the long clearance at seg 9 0:06-0:08;
      between phones it is metric (2 m per frame) plus 1. Under a
      second per clip. Found / wrong, same candidates and filters, the
      chain at this commit -> the path: seg 9 full scans 16/37, 0 ->
      21/37 (57%), 2; win 14/37, 0 -> 18/37 (49%), 2; seg 3 full 36/53,
      2 -> 38/53 (72%), 2; win 36/53, 1 -> 39/53 (74%), 4; coverage
      31/28/70/66% -> 44/40/70/72%. Two of the wrongs on each clip are
      7-8 m from a label 45-58 m away (hit radius 5-6 m). Sweeps: found
      is flat over no-ball cost 2-3, gap penalty 1-3, motion scale 2-3
      deg and skip reach 10-30 frames (only wrong moves, 2-6); a no-ball
      cost of 1 drops short confident runs (the clip's opening track),
      0.75 drops everything; consecutive-only edges collapse to 5/37;
      interpolating the skipped frames (as ball.py's 1 s bridge) is
      worth 6-11 moments; a near-player bonus, a stillness charge and a
      backward smoothing of the positions change nothing. Holding a
      run's end position 0.5 s into the frames around it (HOLD_S) lifts
      found to 24/37, 23/37, 41/53, 39/53 at 3-5 wrong, but that is a
      stale estimate follow.py already supplies itself (ballhold), so
      it is off. Ceiling on the filtered candidates: one within the hit
      radius at the labelled frame exists at 15/37 and 31-32/53
      moments, within +-0.5 s at 27-28 and 41-42; the path's misses are
      9/37 and 10/53 moments with no candidate within +-0.5 s, plus 3-8
      with a lone candidate 1-5 frames off. The remaining gap is
      detection. Play boxes, same run, chain -> path (ballwin
      candidates): A2 11/13 -> 9/13, A2 + roles + kickoff 12/13 ->
      10/13, mean coverage 98-100% -> 96-97%, median width 61-63 deg
      either way; the path aims at the ball in 39% / 70% of the clips
      against 27% / 65%. The two lost boxes are both on seg 9: 1:57
      (67%): the path tracks a slow ball in the left corner at
      1:54-1:56 (judge 1.0 on the secondary phone, 25 m from the goal,
      which the chain never links), and the goalkick rule's far-ball
      stop (GK_BALL_FAR_M) ends the keeper's framing a second before
      the kick; and 0:13 (95%): identical targets, a slightly
      different smoothed camera. So at first the chain stayed the
      default and the path sat behind a flag. Reviewed on the side-by-side render (2026-10-19): the
      path's camera is right at 1:54-1:57, the ball is in the corner
      and a keeper framing need not hold, so the 1:57 box was removed
      from the seg 9 set (12 moments now), and the 0:13 box was redrawn
      to the clearance itself (its old left edge held two bystanders
      the path's camera clipped at 95%). On 12: chain and path both
      11/12, mean coverage 100%, missing only the seg 3 opening (95%);
      the path is narrower at the captures (55 vs 60 deg median) and
      aims at the ball in 39% / 70% of the clips against 27% / 65%. So
      the path is now the default ball linker in follow.py and
      balleval.py (OSB_BALL_CHAIN=1 gives the chain back); ball.py's
      filters and ballwin.py's online track stay as they are (the
      search windows need the online prediction). Still open: the
      goal-kick far-ball stop (GK_BALL_FAR_M) treats a slow ball in the
      keeper's corner as play elsewhere; stopping only on a ball moving
      away would be truer, but no box measures it now.
- [ ] Teams from kit colour (`teams.py`, 2026-10-17): RF-DETR people on
      the clip label sets, torso patches white-balanced against the turf
      beside the player, colour features, a Gaussian mixture of 8. The
      kits come out clean on both sets (blue; solid green; green/white
      stripes; pink referee + pink keeper together; yellow keeper), the
      rest are grass-dominated far torsos, skin and bench. Sides by
      headcount (the two-way split of the kit components with the most
      equal numbers moment by moment; kit components are those with
      >= 2.5 on-pitch people per moment, saturated and tight in chroma):
      seg 9 gives blue vs green + stripes with a 44% margin, which is the
      right answer (the stripes are the green team's second kit). The
      "kit component" criteria are thresholds tuned by eye on one game.
      Wired in (`roles.py` + follow.py "roles"): people along a clip at
      half resolution with side / keeper / referee (keeper vs referee by
      goal-area position; a far, small, saturated person in a goal area
      counts as the keeper). Referees are dropped before tracking; a
      keeper in their area with the ball triggers the goal-kick framing.
      Play boxes 11/13 -> 12/13 (the seg 9 goal kick); keepers seen in
      85-99% of frames. ~12 min per 2 min clip on the Mac (another
      ~6 h on a full game; the stage could share ballwin.py's decode).
      Kick-offs ("kickoff", 2026-10-18): the ball parked at the centre
      spot for 3 s with hardly anyone running, and with roles both sides
      nearly in their own halves, holds the centre circle until play has
      moved off it. The sides alone were not enough: at the first
      kick-off (1:09) the stragglers were still crossing while the ball
      had sat at the centre since 0:18. Fires at 0:46-1:10, 8:54-9:05
      and 50:01-50:12 (the second-half kick-off), not at the ball's
      brief stops at 4:12 and 34:24. Whether 8:54 follows a goal is
      unconfirmed. A keeper alone is no longer a goal kick (they are at
      every kick-off), and no goal kick is framed within 8 s of one.
      Next: the close-pairs rule as a second vote, team as a matching
      cost in the player tracker, keeper bearings as a ball witness with
      the real keeper. When the margin is small, ask once in the app.
      Other games: automatic sides failed on every one since. rec4
      (2026-10-22, red vs white): the split picked two shades of red as
      the two sides, since the saturation floor drops white kits; set by
      eye in `teams/seg9/sides.json` (red vs white, the teal keeper in
      with white, yellow referee). The warm-up chunk's roles are
      lopsided (training gear, not kits). rec5 (2026-10-29, blue vs
      white, soft iPhone half): one kit component (white under the
      saturation floor), "fewer than two kit components"; set by eye.
      Over 10 min roles gave white ~2x blue, so the white group also
      takes subs and spectators. rec7 (2026-11-05, white vs yellow): with
      5 components the two kits merged (white reads yellowish under
      floodlights); with 8, kits [0] and [2] at 10.6 vs 5.8 a moment,
      still wrong by eye; set by hand (A white [0,7], B yellow [1,2],
      keeper/referee [5,6]); on unseen footage (segment 10) white still
      runs low. roles.py now runs without goals.json (goal-line
      centres). The kit-component thresholds need more than one game;
      asking in the app is the fallback.
- [ ] Camera-log review of the second full render (2026-10-18/19,
      `camlog.txt`: one line per change of the rule driving the camera):
      - 25:05 framed a goal kick at the right end during open play at
        the left (the keeper alone + "play near that goal 11 s ago",
        the goal at 24:54). Fixed: a confident raw ball candidate far
        from a goal, with none near it, vetoes a goal kick there (the
        chain often fails to link those candidates). A lull-based guard
        was tried first: no help, and it cost a play box.
      - 54:32 and 54:41 followed the two spare balls by the bench (just
        inside the 3 m pitch margin, seen only when a scan passed, so
        the static filter missed them). Fixed: in loiter cells a ball
        still for 8 s with 3 sightings is static; elsewhere 10 sightings
        over 30 s regardless of gaps. Excluding loiter cells outright
        lost the throw-in box (the thrower stands in one).
      - ballwin.py's online track accepted a 22 m jump, got a 33 m/s
        velocity and flew off the pitch, refusing good candidates for
        1.4 s; in the full run its windows sat on the bench ball for 7 s.
        Fixed: velocity capped (20 m/s), decayed while unseen, prediction
        kept on the pitch, big jumps need conf >= 0.6. Clip recall a wash
        (seg 9 46 -> 38%, seg 3 66 -> 68%), 25:04-25:26 now tracked.
      - 53:42 (a long ball ahead of the players) and 54:02 (a lull hold
        after the keeper gathered) were correct.
      Follow-up: restart_taker flashes 95 deg wide for 0.4-2 s in short
      lulls (54:27-54:50); gate it like the goal kick (far-ball veto,
      minimum lull length). Done differently on 2026-10-23: the taker
      must be where play just was (rec4 follow cam item below). The
      ball-stage changes mean a full re-render
      needs the ball candidates regenerated (~45 min per 10 min chunk).
- [ ] Play boxes, round three (2026-10-19): 13 more reviewed moments
      from the first 23 minutes (sets seg0-0, seg0-600, seg0-1200; the
      page now lists game time for sets that start at the game's start),
      25 in all. The default (A2 + roles + kickoff) captures 19/25,
      players-only 11/25. Of the six misses, four had the ball detected
      but unused: a throw-in (10:02, in hands for 4.5 s, judge 0), two
      single-frame sightings (14:05, 19:36), one short run; two (7:34,
      7:51, the left penalty area 51 m away) have no candidate at all.
      Halving ballpath.py's gap penalty (2 -> 1) lets isolated sure
      sightings join the path: 20/25, and on the ball labels the same
      found-rate with fewer wrong (seg 3: 4 -> 2). Adopted. Tried and
      not adopted: a longer ballhold (5-8 s: loses two boxes to stale
      holds), no judge filter (loses a box and 11 points of recall),
      the path's no-ball cost 1.5 (loses a box).
      Player-movement heuristics (off by default): "surge" (>= 60% of
      >= 4 running players heading the same way along the pitch: aim
      5 m ahead of the frontmost) fires in 19% of frames but moves the
      smoothed aim only 4.5 deg at the median (the hotspot's running
      pool already sits near the front); neutral on the boxes with the
      ball (20/25), +3 without it (14/25 vs 11). Stricter agreement or
      a 10 m lead: no change. Reframed to show the SURGE_FRONT_N
      forward-most non-keeper players plus 5 m ahead (2026-10-19): with
      the ball 19/25 at N = 3 or 5 (the surge view pulls off boxes the
      ball rule had, 19:36 to 76-81%) and 20/25 at N = 7; players only
      14/25 either way (vs 13 with roles alone). A modest fallback where
      the ball isn't tracked, never a gain on top of it; off by default. "sideline" (hotspot within 10 m of the
      near touchline and 25 m of a goal line: frame everyone in that
      corner, padded) fires in 0.6% of frames, since the hotspot drifts
      to the pack even when play is on the touchline; neutral on every
      box, and a wider zone (18 / 40 m) costs coverage at 14:05. The
      trigger would have to come from the ball or the thrower (roles),
      not the hotspot.
      Open at 10:02: the throw-in sits in a judge-rejected gap; a judge
      that accepts a ball in hands, or a thrower rule from roles (a
      player on the touchline with a ball candidate overhead).
      7:22 added (2026-10-20, a counter down the near-left wing): captured
      by the ball rule (two confident sightings 1.5 s apart, bridged by the
      path), 21/26. Surge would also have fired there (10-16 runners,
      82-94% heading left) but the ball was 5-10 m ahead of the front
      five. Surge can't fire at 7:34: only 0-4 players run around it (a
      set piece), but the five forward-most stand at the left goal line,
      x -49..-34, 40 m beyond the hotspot: a positional trigger (the
      forward-most players bunched far from the hotspot) would target
      7:34 and 7:51. Not built.
- [ ] rec4 follow cam (2026-10-22/23; must-see boxes drawn by the user:
      10:00-20:00 six moments, 20:00-30:00 nine; TESTING-PROCESSING.md
      section 2). Pitch and cameras for this game are in the lens and
      pitch-map items.
      - First run (first camera model, rec3's detector and judge): 0/6
        at 10:00-20:00 (unverified); ball stage 62-68 min per 10-minute
        chunk (rec3 42-58). Re-planned with the corrected cameras, the
        h26 pitch and fitted goals: 3/6 (16:54's false goal kick gone:
        off-pitch people by the near touchline had counted as on the
        pitch).
      - The rec3-trained detector loses the ball for 4.5-6 s at a time
        before 16:37 and 17:11; the judge rejects nearly every candidate
        (1 in 30 above 0.5); a spare ball sits by the far touchline.
      - Pass A's motion tracks see 7-12 of the 24-29 people roles.py
        detects; at 17:11 three players breaking toward the right goal
        are untracked. roles.py runs at half resolution: far-touchline
        players are ~20 px tall, 161 detections there in 10 min against
        ~34,700 mid-pitch. Full resolution for the far band not tried.
      - peoplesurge (c57d4d8): surge on roles.py's people (no referees
        or keepers), each with a velocity from the nearest person 0.5 s
        before and after within 4 m (~18 per roles frame). At 17:11 the
        tracks' front sat at x -14..+16, the real front at +40..+44.
        10:00-20:00: 5/6 against 3/6 (16:37 and 17:11 gained; 18:27
        still a false far-left ball); surge chooses the aim in 10.5% of
        frames, ~1 s per chunk. Needs roles; off by default (fullgame.py's
        default features lack it).
      - The overnight full-game run with peoplesurge scored 0/8 on
        20:00-30:00 (28% best); a cold start on 25:00-30:00 alone 5/8
        without peoplesurge, 4/8 with (both unverified). Cause: the loiter
        filter is computed per clip, and its 4 m grid, dilated a cell,
        reached ~7 m into the pitch by the near bench: it dropped 32% of
        detections over 10 minutes and 49% over 5
        (`footage/rec4cal_cmp/follow_*.log`). Replaced (494ffdc) by bands
        along the lines: only detections outside a line or < 0.5 m inside
        (2 m on the far touchline) in a 4 m stretch busy >= 10% of frames;
        ball.py's static spots use the same test. On 25:00-30:00 it drops
        nothing; 25:10 gained, 29:56 lost to a restart "taker" that was a
        leftover background blob 50 m from play.
      - Two balls (fa81780): at 20:25 the path jumped 55 m in 1 s from the
        match ball to a second ball a group in pink bibs kicked in the
        empty half, and the ball rule followed it for 5 s. Runs <= 3 s
        apart (JUMP_GAP_S) whose join needs > 35 m/s keep only the one
        nearer the players. On four rec4 chunks it cut jumps of 55, 75
        and 91 m/s and kept all up to 32 m/s (refined 2026-10-27 with
        JUMP_PLAY_MARGIN_M, above).
      - Restart taker must be where play just was (aa2dff1): play = the
        ball seen in the 5 s before the lull, else the held hotspot; a
        throw-in taker within 20 m of it along the line, a corner taker
        at its end, a goal-kick taker at its end with play within 30 m of
        that goal line. At 25:20 the right keeper, alone 40 m from
        everyone, had been framed as a goal kick while play was at the
        left end. Restart time on 20:00-30:00 30 -> 18 s. Alone it loses
        rec3 seg 9 0:12-0:13 (captured before only because an idle
        keeper's 95 deg framing happened to cover play).
      - Surge frames its runners (0c96b43): the front five of everyone
        included subs on the near touchline and a lone player upfield
        (28:47: three subs 30 m from any runner), so it aimed at empty
        ground. Now it frames the runners heading the surge's way plus
        5 m, dropping the rearmost to fit 70 deg. Alone: rec3 21/26 vs 20
        (13:50), rec4 20:00-30:00 1/9 vs 0 (28:47), 10:00-20:00 4/6 vs 5
        (16:37 at 95%, captured before by a sub widening the view).
      - Each change alone, peoplesurge on (rec3 /26, rec4 10-20 /6, 20-30
        /9): committed 20, 5, 0 = 25/41; loiter 21, 5, 2; jump 20, 5, 0;
        restart 18, 5, 0; surge 21, 4, 1; all four 20, 4, 5 = 29/41. Jump
        and restart only pay off together (20:27 and 25:20 captured only
        with all four). Without peoplesurge 28/41. A full render with the
        four (fullgame_v2, 3.7 h with ball and roles reused) scores the
        same as the bench on both rec4 sets. Still missed: 18:27 (false
        far-left ball), 26:31 (a second ball by the left touchline),
        26:49, 28:04 (29%), 29:56.
      - Scoring caveat: a box counts as captured when 98% of it is
        anywhere in view (16:57 sat in the left quarter). A centred or
        minimum-size test was proposed, not built.
- [ ] Local footage removed (2026-10-21): the phones' segments (48 GB,
      in S3, checked byte-for-byte), both full renders, the clip and
      comparison videos and the spare training checkpoints went to the
      Trash (128 GB). The benches, the tracker scorer and the pitch test
      run without them (docs/TESTING-PROCESSING.md section 5); renders,
      new detectors and new label sets need segments back from S3.
- [ ] Skip what isn't play (`phases.py`, 2026-11-08). `features` makes a
      CPU + hardware-decode pass (Main's audio for whistles; both phones
      at 1 fps / 640 px, motion per pitch cell, NX x NZ = 6 x 3, and
      contact sheets every 15 s); `segment` splits pre / half / break /
      half / post and writes phases/phases.json. Truth by eye in
      phases/truth.json (rec7 whole; rec3 kick-off 1:05; rec5, rec6 begin
      in play). Next:
      - Whole games to test on: only rec7 has all its video on disk (rec3
        0-6, rec4 2-6, rec5 0-2, rec6 0-2 of their segments; ~50 GB per
        game, 23 GB free). Fetch segment by segment and delete, or run
        the pass where the video is.
      - Half ends run late (rec7 +107 s, full time +97 s once kept to the
        end): players walking off and huddling by the mount read as play.
        End-of-half whistles come in clusters (rec7 50:07, 50:16, 50:17
        1.0 s); a cluster score, or the pitch emptying over the next
        minutes, may place them.
      - Whistle pitch on short or quiet recordings: rec4-6 partials chose
        2.2-2.7 kHz. Require a minimum count, fall back to motion alone.
      - Pass cost: decoding every frame for 1 fps is ~22 min a game; try
        keyframes only (`-skip_frame nokey`) once the phones' keyframe
        interval is known.
      - Wire into `fullgame.py` as ranges to render (it has only
        `--start`), padded (60 s used here) and never cutting inside a
        half.
      - Any number of periods, and breaks that have to be shown
        (2026-11-10). `split` was five states (halves 20-55 min, one
        break 2-25 min) over a play score scaled between each recording's
        own 10th and 90th percentile. It is now a DP over any number of
        periods (up to 8, each 8 min or more, breaks 2 min or more, no
        upper limits) where every stretch that isn't play (pre-game, a
        break, post-game) costs BREAK_COST = 20 bins of full play score,
        and a break between periods is scored against a lower threshold
        (THR_BREAK: the pitch empties; a warm-up or a lull only slows).
        The play score is motion as a share of the recording's 90th
        percentile with 0.45 of it as zero (rec7's own 10th percentile is
        0.46 of its 90th, so rec7 scores as before; rec3-rec6, all or
        nearly all play, are 0.40-0.59, which is why their lulls read as
        breaks). `refine` runs for exactly two periods and only moves a
        kick-off earlier or an end later, so the equal-halves prior can't
        shorten a half to match one the recording cut. phases.json gains
        "periods" (start, end, whether each was seen, its whistle) and
        "play" (all / some / none); "found" keeps k1/e1/k2/e2 for one or
        two periods; "refereeHz" is null, not NaN, when no whistle
        qualifies; a Main file with no audio track is split on motion
        alone (it crashed in ffmpeg).
        Real recordings, old -> new: rec7 9:01 / 52:04 / 56:14 / end both
        (errors +1, +107, +1, +97 s; no play lost; 9-10% cut). rec3 (33
        min on disk) ended play at 23:10, now keeps all of it: the
        contact sheet shows play at the far end until the recording
        stops (read by eye, 2026-11-10; truth.json has only the
        kick-off). rec4 (segments 2-6) cut 9:29-14:30, now keeps it:
        quiet, probably before kick-off, no truth; a loss of 5 min of
        saving if so. rec5-game, rec6: all kept, both.
        Made-up recordings (motion at a play or a quiet level, 0.9 s
        whistles at the changes; `processing/tests/test_phases.py`),
        old -> new: extra time (45+45, then 15+15) all 30 min after the
        game -> four periods; two games of 2 x 30 on one recording, 11
        min cut from each -> four periods; one 60-min period given a
        3-min half-time -> one period; a 12-min stoppage ending the half
        2.5 min early -> a break; 4 x 12 quarters merged in pairs ->
        four; 40 min of play given a 7.5-min half-time -> all play.
        Limits, each erring toward keeping: on motion alone (no
        whistles) a break needs ~5 min of empty pitch (2-min quarter
        breaks and a 2-min extra-time break merge; a 5-min half-time
        between 15-min halves is missed); under 8 min of play at the
        edge of a recording next to a break isn't a period, so a
        recording that starts or stops within 8 min of a break loses
        those minutes; a recording with nobody on the
        pitch throughout reads as all play (the score has no absolute
        level: rec7 plays at 0.026 of pitch pixels moving, rec3-rec6
        at 0.055-0.065); more than two periods get no whistle
        refinement (rec7's second kick-off was 3 min late without it).
        How sure (rec3, rec4, rec7): no setting tried cut rec7's play
        (BREAK_COST 10-40, QUIET_FRAC 0.35-0.55, THR 0.4-0.6, THR_BREAK
        0.1-0.5, MIN_PLAY_S 5-20 min; the worst is its second kick-off
        53 s late, inside the 60 s pad). rec7's pre-game is cut only at
        BREAK_COST 10-20 and QUIET_FRAC 0.45-0.55; rec3's false
        post-game returns at BREAK_COST 10 or QUIET_FRAC 0.55. Still one
        whole game.
- [ ] Full-game runner (`fullgame.py`): name each piece by the
      configuration it was rendered with (a hash of the feature string
      and the constants it uses, plus the candidate and roles file
      versions), so a later run re-renders only the chunks whose inputs
      changed; today a changed rule means re-rendering the whole game
      (~5 h with the ball candidates reused). Also merge roles.py into
      ballwin.py's decode (both read the same frames; from scratch the
      two stages are ~68 min per 10 min chunk). Two hangs seen at the
      end of the last segment file (the decoder at EOF): the last chunk
      now stops 1 s short, and the stages exit hard after writing.
      rec5 whole game (2026-10-30/31, 92.3 min, rec3 + rec5 models, sides
      by hand): 11.0 h on the Mac, ball stage 44-82 min per 10-minute
      chunk. Aim at the ball by chunk 36, 45, 39, 29, 36, 31, 23, 32, 25,
      27% (against 3% on the first 10 min with the night-game models).
      Weaker after the hour (light fading on the orange ball, or
      scrappier play: not checked). No play boxes for rec5, so the share
      is coverage, not accuracy.
      rec7 (2026-11-05, 95.2 min from segment 1, standard models): ball
      stage 70-75 min per 10-minute chunk, follow 27-29 min, ~2x
      rec4cal's pace (cause unchecked: the iPhone half, or the ball lost
      more and scanned more). With judge_v4 the ball
      stage fell to 38-51 min (a held track needs fewer full scans, 6.5
      vs 9.2 tiles per frame). labelset.py clip and playeval.py make had
      read Secondary from the file with Main's segment number; with a
      300 s iPhone that is often the wrong file. Both now use
      follow.phone_at (1a12e90).
- [ ] Follow cam: clip further in from the left and right edges. The edge
      rule stops at the canvas boundary (where a phone's frame ends, less
      2 deg), but the last several degrees of each phone are poor: lens
      fall-off and vignetting, the radial distortion's fold-back limit
      (`Camera.r2_max`), softer focus at the frame corners, and the
      vertical edges of the panorama are the frame corners' diagonals.
      Measure where the usable image ends (sharpness and brightness per
      column near each edge, and r2_max in degrees), set the margin from
      that, probably per side (the phones differ: focal 1507 vs 1485 px,
      the Secondary mounted upside down), and check with a render of a
      near-edge moment (seg 3 at 0:25-0:55 right, 1:40-1:55 left). The
      zoom floor at the edge can stay at 40 deg: pixels per degree are
      highest there.
      Next: tiny far-side balls are the limit. Fine-tune the detector
      (not the judge) with the clip labels as extra tiny-ball examples
      (cross-clip to measure), try 3-4x zoom windows for the far third,
      and plan more pixels per degree at capture. Also a restart rule for
      takers in loiter cells, and more labelled moments from other
      stretches before tuning further.
- [ ] Pitch map (`pitch.py`): background median, white-paint mask, fit
      of camera position/height, pitch rotation, ground tilt, length and
      width, plus a smooth vertical-angle correction (`warpDeg`, what one
      flat pitch and the camera model can't explain). Goals (`pitch.py
      footage/rec3 goals`, once per game): goal-shaped white structures
      near each end are candidates; each left/right pair is tried by
      refitting so they stand mid goal line; the pair that keeps both
      ends' goal line and 6/18-yard boxes on the paint wins (stored goals
      and other pitches' fail). The goals then correct the map: the right
      end was ~20 m off before. Night game: 109.3 x 60.5 m, camera 2.7 m
      up, goals 7.6 and 7.4 x 2.44 m. Still one-off parts:
      - The starting pose was hand-read from the panorama (a guess can be
        passed as pitch.py's third argument since 845c4a5). Since
        2026-11-05 (1a12e90) pitch.py tries START_POSE at camera heights
        1.9-10 m and keeps the plausible fit that lies best on the paint:
        rec7 6 m (+0.39, the others <= +0.16), the same fit as a hand
        start; rec6 4 m, 105.0 x 71.4 m, nearer its taps fit than 1.9 m's
        84.9 x 61.3. map_pitch (the fixtures) still starts at 1.9 m. Still
        to automate from the centre circle ellipse + halfway line, with a
        tap-two-points fallback in the app (centre spot, a corner).
      - Assumes a regulation layout and full-size goals; youth and
        small-sided pitches need a pitch type or more free sizes. The
        lines alone don't fix the scale: with the markings' size free the
        night game fits equally well as 88 x 43 m with markings x0.61
        (5.0 vs 4.8 px; unverified), so the regulation 9.15 m circle sets
        it (ec9cefb). A youth pitch with scaled markings needs the scale
        from elsewhere (gravity tilt, a tapped distance, or goal size).
      - Thresholds tuned on one game; every other field so far needed
        taps (below).
      - Goals moved mid-game (portable) would need a per-segment re-check.
      rec4 (2026-10-22): from rec3's START_POSE the fit settles in a wrong
      minimum (98.7 x 62.7 m, the near touchline off); a start guess gives
      100.7 x 61.1 m at 2.09 m up, but with the camera 0.7 m behind the
      near touchline (really ~3 m) and the model touchline off the
      panorama's edges. The far paint was mostly lost by the mask (far-band
      line pixels 10,200 against rec3's 53,500; unverified): thin far lines
      stay saturated (S < 60 dropped them) and Main's colour cast put far
      grass at hue 25-30, outside turf() (> 30). line_mask now uses S < 100
      and hue > 20 (637b3ba; this is what broke the regression test,
      below). Goals detected as 6.0 x 2.0 and 5.2 x 1.7 m, rejected. The
      painted near touchline is a great circle on Main's half (0.03 deg
      rms) but bends 1-3 deg on the Pro's.
      Taps (2026-10-22, 8c77fbf; the pitchtap page, 12 at the ends then 15
      with the centre): automatic junctions found only the right 18-yard
      box, so the ends come from taps. With them the ends are right but
      the near touchline falls inside its paint; one height offset for the
      whole near touchline explains it (-39 cm, rms 3.54 -> 0.24 deg, both
      phones within 3 cm; not lens, not a crown: wrong sign). The mount was
      measured at 2.6 m. Height held at 2.6 m ("h26"): 105.5 x 62.3 m,
      markings 3.4% large, near touchline and ends right, centre circle
      ~19 px (0.6 deg) off; a crown term (+6 cm) doesn't fix it; heights
      2.6-2.9 m trade circle for ends. With all 15 taps and height free:
      3.11 m, circle and ends right, near touchline off. No flat pitch
      holds both: the near touchline sits 2.6 m below the camera, the
      centre and ends 3.1 m. h26 is used (follow.py has no curvature).
      Goals fitted to the white posts from the tapped post base: 7.67 x
      2.64 and 7.78 x 2.68 m, so h26's scale at the ends is 5-9% small
      (the night-2026-10-21 fixture README and baseline_scores.txt).
      Other fields need taps too (2026-10-29/31). rec5 (yellow paint,
      portable spare goals): lines-only fits run off; with 11 taps 88.3 x
      59.3 m, camera 2.27 m up, most taps within 0-10 px but the right 18-
      and 6-yard corners 44-50 px off (unverified): those boxes are
      shallower than regulation and the model scales all boxes and the
      circle by one markingScale. Goals not run (`pitch.py goals` would
      overwrite the taps refit, and spare goals stand along the far side).
      rec6 (daylight): the lines-only fit diverged (camera 0.94 m up, 84.9
      x 61.3 m); from taps 104.7 x 68.9 m, camera 2.43 m up, after swapping
      two circle taps the page had recorded the wrong way round. The near
      touchline still sits a little outside its paint at the bottom of the
      panorama.
      rec7 (2026-11-05, camera ~6.2 m up after the taps refit): from START_POSE (1.9 m) the
      lines-only fit ran off (camera 0.58 m up); the multi-height start
      above fixes that. With 15 taps and w3 (a depression term in the
      warp, fbf3655): near touchline within 0.27 m (was 0.5 m), worst tap
      21.5 -> 16.1 px, mean 7.5 -> 6.6, camera 6.6 -> 6.2 m, w3 6.2 deg
      (expected ~1; it trades with the camera height, so it is not a lens
      measurement). Free in the lines-only fit, w3 ran the pitch to 136 x
      69.5 m, so it is fitted only with taps. Fit 102.3 x 60.5 m. Goals:
      frames found at x +-50.8 m, both rejected (end lines 10.0 / 4.9 px
      off); the fallback goal-line centres are within 0.35 m, so left as
      is.
      Regression test: `processing/tests/test_pitch_regression.py` reruns
      the whole mapping on saved background panoramas and checks goals,
      size, camera and 15 key points (<= 0.1 deg) against results checked
      by eye. Fixtures hold footage, so they stay out of git
      (`footage/fixtures/pitch/<name>`, made by `pitch.py <rec> fixture
      <dir>`; back them up privately). One so far: the night game
      (`night-2026-10-14`). It has failed 4 checks since 637b3ba
      (2026-10-22): the right goal 7.8 m wide against 7.4, length +1.1 m,
      lines-only key points up to 0.37 deg; the fixture's left goal was
      updated to the new mask (7.6 -> 7.4 m), the right kept until the
      right goal is sorted out.
      Coloured paint (2026-10-29, b74f2a1; rec5: football in yellow beside
      red and blue lines for other sports): colour_mask finds paint of any
      colour in PAINTS by its Lab direction against the turf;
      pitch/paint.json sets it, white without one (the night game's white
      mask bit-identical). Under floodlights the far yellow paint differs
      from the haze by 1-3 b* units, so only the near touchline and most
      of halfway are found, and a yellow lines-only fit runs off (202.7 x
      49.2 m, camera below the ground). Choosing the colour by which
      finder fits best picked white on all three games, wrong on rec5
      (white catches pale far yellow and floodlight noise); scoring only
      the near markings tied on two games. What works: take the best
      plausible fit of any colour for where the lines are, then measure
      the paint under its near markings against the turf beside it: b*
      -10 (rec3) and -12 (rec4) -> white, +3 (rec5) -> yellow (cut-off +1:
      a small margin). With yellow and a hand start pose the fit converges
      (86.2 x 54.4 m, camera 2.0 m). The tap page shows the choice for
      confirmation. Orange, red and blue are untested on real fields.
- [ ] Ball labelling: `labelset.py` (300 random moments, RF-DETR
      suggestions, stitched panoramas; unlabelled frames turned into
      both-phone moments) and `label/server.py` (local page: panorama
      plus a magnifier per phone; every ball marked, the match ball
      flagged; repeat figments hidden after the first few labels).
      408 of 409 frames labelled (183 with the match ball).
- [ ] Ball detection (night game, 2026-10-15):
      - Off-the-shelf RF-DETR "sports ball" on 4K tiles: 58% of match
        balls at conf >= 0.2 with 2.1 wrong suggestions per frame.
      - Motion (`motion.py`: difference from the median of +-1 s and from
        the neighbouring frames, blobs of ball size for the distance):
        55% of match balls but ~120 candidates per frame, and ranking by
        contrast is poor (top 10: 6%). Detector + motion together: 77%.
        Decode natively (`-fps_mode passthrough`): ffmpeg's frame-rate
        filter repeats the first frame after a seek, hiding all motion.
      - Fine-tuned RF-DETR Medium (`balldata.py` 640 px COCO tiles with
        hard negatives, `balltrain.py`, batch 2 x 8 accumulation on mps,
        ~7 min per epoch, early stop at 26, best epoch 18; mAP50 0.64).
        On held-back moments (`ballcompare.py`, 42 match balls): 76% of
        match balls at conf >= 0.1 with 1.1 wrong per frame (stock: 69%
        with 10.5); the top suggestion is the match ball 69% of the time
        (stock 40%). On training moments 96%: overfit to this game, so
        label a second game before training more. Its confidence means
        something: ~60% of match balls still found at 0.4-0.6, 57% at 0.7
        (stock 36%); it finds fewer of the spare balls (all balls 73% vs
        81% at 0.1). Unverified (transcript output).
      - Next: the tuned detector as a judge of motion candidates, and a
        tracker with ball physics (ground rolls, lofted balls projecting
        long).
- [ ] Ball and roles stage speed, from a profile (2026-11-06). One
      10-minute chunk (rec4 0 1800 600, `fullgame.py`'s arguments: ball
      at 10 fps, roles `seg9` at 4 fps), run on a scratch copy of the
      recording with GPU-synchronised timers wrapped around the spikes
      (no source changes). An App Store install was using CPU during
      the run; the stages are GPU-bound.
      - ballwin.py, 42.4 min (Oct 22's run of the same chunk: 39.7 min at
        5.9 crops per frame pair, now 8.7). RF-DETR 34.3 min (network
        30.8): 52,106 crops in 12,746 calls. Full scans (42 crops per
        frame pair, 984 of them) are ~41k of the crops; windows and zooms
        ~11k, mostly one crop per call. NV12 -> RGB of the whole 4K frame
        on the GPU 4.5 min (~23 ms per call); the frame copied to the CPU
        for judge crops 2.6 min; the judge itself 0.2 min; waiting for
        decode 0.02 min.
      - roles.py, 23.8 min: RF-DETR (stock Medium, 8 crops of the
        half-resolution frame per phone) 22.6 min; teams.features 0.6
        min; CPU colour conversion and resize 0.1 min; decode wait
        0.02 min.
      - follow.py could not run (rec4 segment 0 is no longer on disk; the
        exposure match reads it). rec7's chunks of 2026-11-05: ~25 min per
        10 minutes, pass B render ~880 s (48 ms/frame), pass A warp+blend
        ~320 s, seam ~150 s, background ~90 s, encode ~28 s.
      - RF-DETR network per crop on MPS (random input, after warm-up):

        | Model | batch 1 | batch 8 | batch 16 | batch 8 fp16 |
        |---|---|---|---|---|
        | Medium (576) | 46.0 ms | 32.5 | 30.7 | 28.1 |
        | Small (512) | 38.3 | 24.9 | 23.5 | 21.6 |
        | Nano (384) | 24.6 | 14.1 | 13.1 | 12.6 |

      Ideas, each to be checked on the play boxes and ball labels:
      - Fewer full scans: the lost-ball rescans (every 0.5 s) fire for
        much of the clip, and scans are ~79% of the detector's crops. The
        4 s / 1 s cadence already exists as env settings; a cheaper lost
        scan (only the band where the ball can be, or the last-seen area
        first) is the larger lever.
        Benched 2026-11-07 on all 46 must-see moments (rec3, rec4, rec7),
        fresh candidates per cadence (`OSB_FULL_EVERY_S` / `OSB_LOST_SCAN_S`):

        | Full / lost scan | Captured (rec3/rec4/rec7) | Ball stage, 8 clips | Full scans | Labels seg3 / seg9 |
        |---|---|---|---|---|
        | 2 s / 0.5 s (today) | 33/46 (21/8/4) | 235 min | 4,386 | 81% / 62% |
        | 4 s / 1 s | 32/46 (19/9/4) | 178 min (-24%) | 2,365 | 75% / 62% |
        | 2 s / 1 s | 32/46 (19/9/4) | 195 min (-17%) | 2,834 | 77% / 59% |
        | 2 s / 2 s | 32/46 (19/9/4) | 166 min (-29%) | 1,920 | 72% / 57% |

        The same moments change in all three: rec3 14:05 lost (100 -> 0),
        rec3 7:51 lost (100 -> 95), rec4 18:27 gained (0 -> 100). At 14:05
        the ball is kicked ~45 degrees across at 14:03; the 0.5 s rescan at
        14:04.5 finds it, the 1 s rescans each find a parked ball by the
        sideline (yaw 84, judge 1.0) and the windows lock onto that until
        14:09. Next: 4 s full scans with the lost scan kept at 0.5 s, to
        see how much of the speedup survives without that loss; and keep a
        static, off-play ball from winning the lost scan.
      - [x] Smaller full scans, planned from the pitch map (2026-11-07,
        now `ballwin.py`'s default). 93% / 77% / 61% of rec3 / rec4 / rec7's
        accepted balls sit in the top tile row (y 760-1100, 6-11 px wide);
        below y 1400 they are 24-73 px wide but cost 14 of the 21 tiles.
        The plan: the fewest 640 px tiles covering every on-pitch pixel
        where the expected ball is under 20 px (`OSB_SCAN_HALF_PX`), the
        fewest 1280 px crops shrunk to 640 covering the rest of the pitch,
        nothing off the pitch. rec7's phones see the ball small further
        down the frame, so it keeps more full-size tiles.

        | Scan | Cadence | Crops per scan (rec3/rec4/rec7, both phones) | Captured | Ball stage, 8 clips | Labels seg3 / seg9 |
        |---|---|---|---|---|---|
        | grid | 2 s / 0.5 s | 42 / 42 / 42 | 33/46 | 235 min | 81% / 62% |
        | pitch map | 2 s / 0.5 s | 22 / 19 / 33 | 34/46 | 183 min (-22%) | 83% / 65% |
        | pitch map | 4 s / 0.5 s | 22 / 19 / 33 | 33/46 | 176 min (-25%) | 81% / 65% |

        Per recording at 2 s / 0.5 s: rec3 -21%, rec4 -30%, rec7 -10%. No
        moment got worse; rec3 seg3 5:30 rose 0 -> 100 and rec7's second
        moment 33 -> 80. With 4 s full scans rec3 7:51 fell 100 -> 90 for 3
        more points of speed, so the cadence stays at 2 s / 0.5 s. Most
        scans are now lost-ball rescans; a cheaper rescan (near the last
        sighting first) is the next lever.
      - [x] ~~Local lost-ball rescans~~ (2026-11-07, tried, not adopted;
        prototype on branch `ballwin-local-rescan`, `OSB_LOST_LOCAL=1`).
        With the pitch-map scan, the ball is seen in 36% of frames, missing
        under 1 s in 26% and lost (>= 1 s) in 38%; 77% of full scans are
        lost-ball rescans (rec4 20:00-30:00: 93%). The prototype limited
        those rescans to the crops seeing pitch within R0 + V x seconds
        lost of the last sighting, with a whole scan still every 2 s.

        | Rescan | Clips | Local scans | Ball stage | Captured | Labels seg3 / seg9 |
        |---|---|---|---|---|---|
        | whole (default) | 8 | - | 10,959 s | 34/46 | 83% / 65% |
        | local, 10 m + 10 m/s | 8 | 2,557 of 4,360 | 11,020 s (+0.6%) | 34/46 | 83% / 65% |
        | local, 5 m + 5 m/s | first 4 of 8 (stopped) | 634 of 1,314 | 3,792 s vs 3,878 (-2%) | - | - |

        Every moment's coverage matched the default. Tiles per frame pair
        fell 5-9%, but time did not follow: after the pitch-map plan, scans
        are only ~half the detector's crops (~1.9 of ~3.9 per frame pair;
        the rest are the tracking windows), and per-frame work (decode, the
        whole-frame NV12 -> RGB, the 4K copy to the CPU for the judge, the
        windows) now dominates at ~0.28 s per frame pair. Further cuts to
        the scans have little left to give; the NV12 crop and GPU judge
        crops above are the next speed levers.
      - roles.py on RF-DETR Small or Nano in fp16 (~1.5-2.6x faster
        detector), or at 2 fps.
      - Done (2026-11-08): ballwin.py converts only the windows, tiles
        and judge crops from NV12 (`detect.nv12_crop_to_rgb`, each region
        with a 1-pixel chroma margin so it matches the whole-frame
        conversion bit for bit), and the judge's crops are copied to the
        CPU as small patches (`balljudge.crops_from`) instead of the 4K
        frame. Default; `OSB_CROP_RGB=0` for the old path. The ball
        output is unchanged: 1,078 of 1,078 judge crops identical to
        cutting them from the whole frame, and the candidates of rec3m
        3:45 and 1:00 (2 min each) identical to the pitch-map default's,
        so the must-see bench is unchanged (34/46) by construction. Time:
        those two clips 524 s against 687 (-24%); the 2-minute rec3m
        10:00 profile 3.9 min against 6.2 (-37%): NV12 -> RGB of the
        crops 0.18 min, judge crops 0.03, and the detector's network now
        77% of the stage.
        Tried first: the judge's crops cut and resized on the GPU (INTER_AREA
        as weight matrices, as MPS has no area pooling at uneven sizes).
        Judge scores within 0.021 of the CPU's (3 of 1,711 crops crossed
        0.3), the ball stage -20% over the 8 bench clips (8,730 s against
        10,959) but 32/46 captured against 34. The accepted ball differed in
        only 54 of 20,400 frames, yet the camera follower carries state
        forward: a 5 s difference at rec3m 10:03 changed the camera path
        from 12:00 on (up to 60 degrees; the follower itself is
        deterministic) and lost 14:05 (100 -> 0), with 7:51 100 -> 95 and
        10:02 10 -> 38. Not adopted; the exact patches are faster anyway
        (4.3 min on the profile clip against 3.9). The bench's single
        moments are fragile to tiny ball changes, so compare variants on
        captures over all 46, not moment by moment.
      - All of one frame pair's crops in one `predict` call (both phones,
        window and zoom); a small gain, as the scans are already batched.
      - fp16 for the ball detector (~10-14%); a smaller ball detector is
        riskier, as Nano's 384 px input shrinks far balls further.
      - The ball detector exported to ONNX and Core ML (2026-11-08, M2 Pro,
        macOS 27.0.1, rfdetr 1.11.2, torch 2.15, onnxruntime 1.31.1,
        coremltools 9.0; script, logs and exported models in
        `footage/onnxbench/`). All five exports worked first time
        (`model.export`, 3-12 s each). Network only, random input, ms per
        crop (3 warm-up + 12 timed calls, as in the table above):

        | Backend | batch 1 | batch 8 | load |
        |---|---|---|---|
        | PyTorch MPS fp32 (today) | 44.9 | 31.5 | - |
        | PyTorch MPS fp16 | 43.1 | 27.2 | - |
        | Core ML fp32, GPU | **25.8** | **21.8** | 0.5 s |
        | Core ML fp16, GPU | 23.0 | 18.1 | 0.4 s |
        | Core ML fp16, all units / Neural Engine | 29.6-29.7 | 33.5-33.9 | 4-24 s |
        | Core ML fp32, Neural Engine | 79.2 | 77.6 | 0.7 s |
        | ONNX Runtime, Core ML provider (GPU or all) | 31.0 | 26.5 | 27-44 s |
        | ONNX Runtime, Core ML provider, Neural Engine | 84.8 | 85.7 | 28 s |
        | ONNX Runtime, CPU | 180.8 | 181.7 | 0.3 s |

        Parity on the 451 tiles of `balldata_rec3_rec5_rec6/valid` (113
        balls), every backend's raw outputs decoded the same way at
        FLOOR 0.1 and compared with PyTorch MPS fp32 (74 / 58 / 51 balls
        found at 0.1 / 0.3 / 0.5; the top detection is the ball on 66 of
        71 tiles): Core ML fp32 and every ONNX Runtime backend give the
        same 150 detections with the same confidences (to 4 places).
        PyTorch fp16 matches 149/150 (confidence within 0.009). Core ML
        fp16 drifts: 133-144 of 150 matched, up to 13 new ones,
        confidence off by up to 0.63, balls found at 0.5 down to 47-50.
        So Core ML fp32 on the GPU is the candidate: 1.74x faster at
        batch 1 (the tracking windows), 1.44x at batch 8 (scans, roles),
        same answers. ONNX Runtime's Core ML provider runs 940 of 946
        nodes in Core ML but is slower than native and recompiles on
        every load (its model cache was not tried). The Neural Engine is
        no use for this model. Estimated effect from the per-frame
        numbers above (~1.9 scan and ~2 window crops per frame pair at
        ~0.28 s): ball stage about -20%, roles about -30%. Not measured
        in the pipeline: crops would leave PyTorch as CPU arrays (8 x 576
        x 576 x 3 floats is 32 MB a batch; unified memory, but a copy and
        a sync per call), so check it in ballwin.py before adopting.
        Checked in ballwin.py (2026-11-08): `OSB_DETECTOR_COREML=<folder
        of the b1 and b8 fp32 exports>` (run with `--with coremltools`)
        swaps the detector for Core ML, batch 8 for calls over 6 crops,
        batch 1 otherwise. On the 451 tiles in ballwin-sized calls it
        matches `predict()` 150/150 (confidence within 0.00003), 29.8 vs
        48.8 ms per crop including preprocessing and the copy. On the
        must-see clips, same settings as the crop variant, back to back:
        rec3 seg 3 237 -> 157 s, rec3 seg 9 301 -> 227 s, rec4 10:00-20:00
        1355 -> 888 s (detector 21.9 -> 13.9 min); the ball stage is
        25-35% faster overall. Candidate files hold the same values,
        apart from 36 of ~104k that differ in the last rounded digit
        (up to 0.01); play boxes 33/46 and seg3 / seg9 labels 44/53 and
        24/37 for both.
        Roles too (2026-11-08): the wrapper moved to
        `detect.CoreMLDetector` and also takes uint8 RGB tiles;
        `OSB_PEOPLE_COREML=<folder of the stock model's fp32 exports>`
        (footage/onnxbench/people_fp32) in roles.py. On the 451 tiles in
        calls of 8 at roles' 0.4: all 2,253 detections (1,771 people)
        match, same classes, confidence within 0.001; 28.8 vs 40.4 ms a
        tile (always full batches of 8, where PyTorch is at its best).
        On the must-see clips: rec3 seg 3 730 -> 500 s, seg 9 679 -> 481
        s, rec4 10:00-20:00 at 4 fps 1346 -> 949 s (detector 21.6 -> 15.0
        min), 29-32% faster; the same people (113,837 both ways) and the
        same role tallies apart from one person in seg 9 (B -> other);
        play boxes 33/46 for both. Not yet tried: a full game, the
        render running alongside.
        Now the default on a Mac (2026-11-08): `detect.coreml_detector`
        exports fp32 on first use (~10 s a batch size) to
        ~/Library/Caches/opensportsball/coreml, keyed by the checkpoint
        (size, mtime) and rfdetr's version, and falls back to PyTorch
        with a message if anything fails. `OSB_DETECTOR_COREML` /
        `OSB_PEOPLE_COREML` = "0" keeps PyTorch, a folder uses the exports
        there. coremltools is in both scripts' dependencies on macOS
        only. Checked with plain `uv run` on a 20 s clip: ball
        candidates byte-identical to PyTorch, roles the same people and
        roles; a bad folder falls back.
        The ONNX Runtime CPU figure (~5.8x slower than the GPU) is a first
        hint for a machine without a usable GPU.
      - Peak memory (2026-11-07, rec7 2-minute clip, `footprint` sampled
        every 2 s, GPU included): ball 5.2 GB (Python 4.3, GPU 3.1; each
        ffmpeg decoder ~0.6), roles 3.4 GB (GPU 1.2), render 3.3 GB (GPU
        1.0); ball stage and render overlapped 8.4 GB, with the render 2x
        slower (89 -> 171 s) and the ball stage 9% slower. Levers if a
        smaller cloud GPU needs them: smaller detector batches, fp16, the
        NV12 crop above, running the stages one after another. The render
        keeps per-frame data, so check its peak on a full 10-minute chunk.
- [ ] Follow cam speed, further ideas (a full game is ~2.5 h locally
      today, half in each pass):
      - Profile (2026-11-08, rec7 0 2084.88 600, follow.py alone on the
        M2 Pro, GPU-synchronised timers patched in from outside, no source
        changes): 18.4 min for the 10-minute chunk, against ~25 min in
        fullgame.py next to the ball stage (pass B 28.3 ms a frame alone,
        44.6 ms there). Pass A 524 s: warp+blend 300 s (16.1 ms, of it
        the uploads 2.2), seam 134 s and background 60 s on the CPU,
        blobs 31 s. Path 44 s. Pass B 531 s, per frame:

        | Step | ms | share |
        |---|---|---|
        | Blend: weights 2.2, two-band lowpass 3.9, blend 3.5 | 9.6 | 34% |
        | Grids: project, upsample to full size (3 channels, fp32), normalise; both phones | 8.5 | 30% |
        | Upload two 4K NV12 frames | 4.0 | 14% |
        | Geometry (directions, seam offset) | 2.0 | 7% |
        | `grid_sample`, both phones, Y and UV | 1.5 | 5% |
        | Gain 1.3, pack 0.7, copy to CPU 0.6, encoder pipe 1.2 | 3.8 | 13% |

        The sampling itself is cheap; the time is many full-frame (1080p)
        elementwise steps, each a pass over GPU memory. First tries, in
        PyTorch: upsample the coarse grid in fp16 with two channels and
        the validity mask at coarse size; the lowpass from a smaller
        image; fewer intermediate tensors in the blend. Then one fused
        kernel per pixel (Metal on the Mac, the PyTorch path kept for
        CUDA), with the decoder's frames used in place (IOSurface) rather
        than copied up.
      - Pass A (detection, seam) at 10 fps, reusing results for the frames
        between: about a third of its cost. Check that the seam's player
        margin still covers fast play near the overlap.
      - Pass A's panorama: scale on the GPU in the decoder (`scale_vt`)
        instead of in software; download it smaller (luma only, or half
        size) or move background subtraction onto the GPU.
      - Seam search in compiled code (numba) or on the GPU instead of a
        Python loop over rows.
      - Fixed mount: precompute a direction -> source pixel lookup once per
        game, so per-frame geometry is a lookup into it.
      - Fuse the render's elementwise steps (`torch.compile`, or a custom
        Metal/CUDA kernel); render in fp16 throughout.
      - Run passes A and B concurrently (B lagging A by the path's
        smoothing window), so decode and GPU work overlap across passes.
      - In the cloud: NVDEC/NVENC and CUDA on the Batch GPU instance, and
        split a game across instances by segment.
- [ ] Clap test on the mount: confirm audio sync also aligns video (each
      phone's audio-to-video offset in the file is assumed equal).
      Indirect check (2026-10-15, unverified): matching motion across the
      overlap agreed with the audio offset within ~1 frame where it
      correlated clearly (4 of 9 probes), noise elsewhere. Capture-side,
      untested: the phones expose up to half a frame apart and the
      upside-down phone's rolling shutter runs the other way (60 fps or
      both phones the same way up would shrink both).

## Phase 3: processing pipeline

- [x] Engine commands for the Mac app and the cloud job (`engine.py`,
      main repo `docs/MAC-APP-PLAN.md` step A), 2026-11-09: `prepare`,
      answers as files, `run`, `status`; platform choices in `accel.py`.
      rec7 from a fresh copy, with a stop and resume, to `game.mp4` with
      no hand edits (FINDINGS section 7).
- [x] The engine for a script or a model with no app (main repo
      `docs/MAC-APP-PLAN.md`, "Using it without the GUI"), 2026-11-10:
      every question says in words what is asked and how to answer
      (`ask`, `howToAnswer`); each pitch keypoint has a close-up with the
      fit drawn on it and each kit group one contact sheet; `questions`
      prints what is pending as one document, `answer` writes an answer
      from flags (a keypoint as a pixel of its close-up, groups by
      number), checked against the question, with sharing off unless
      asked for; `status` names the next command
      ([ENGINE-AGENTS.md](ENGINE-AGENTS.md)). Questions written from
      rec7's pitch and kit groups read well as pictures; unit tests only
      otherwise.
  - [ ] Not yet tried: a model answering a whole game's questions
        unaided (does it mark keypoints within a few pixels? does it
        judge the kit groups as a person does?), scored against the
        answers given by hand on rec4, rec5 and rec7.
- [x] Engine side of the checks and contributions (step B), 2026-11-09:
      `label ball` / `label moments` sets from a finished game, label
      answers stored as `labels.json` / `boxes.json`, `share` lists per
      result. On rec7: 50 ball frames across all 10 chunks, a moments set
      with the camera's view at each moment, four share lists (pitch
      10 MB, kits 417 crops 1.6 MB, ball 8.6 MB, moments 2.9 MB).
- [ ] The standard judge doesn't know rec7's yellow ball: run from a
      fresh copy with the standard models, the camera aimed at the ball
      31% of chunk 0 and 0-2% of every other chunk (`ballAimShare`
      0.035); rec7's own judge_v4 gave 28-64%. The detector still finds
      the ball (6 of 8 sampled no-judge suggestions are on it); the
      judge rejects it. Bench judge_v4 (rec3 + rec5 + rec7) as the
      standard on rec3, rec4cal and rec6. The app now offers ball labels
      early (2026-11-10): `run` emits a `ball` event per followed chunk
      with `ballLow` from the second (under 10% over the chunks after the
      first; a threshold from this one game, to check on more), `status`
      lists each chunk (`chunkDetails`), and `label ball` works while the
      game is still running. `label ball` now fills a set
      from the detector's candidates without the judge when the ball
      path is thin (`judged: false` in the set).
- [ ] Pitch check: the keypoints the user can mark are all on the far
      side and the ends (corners, boxes, halfway far end, circle, post
      bases), so a wrong near touchline can't be corrected. On rec4,
      after the taps the far side sat on the paint but the near
      touchline was ~1.7 deg off under the camera (the field drops away;
      no flat fit holds both). Add near-side points (the halfway line's
      near end, the near corners) and the curvature term.
- [ ] Engine on rec4 and rec5 (step B's "done when"), 2026-11-09, fresh
      copies, first chunk rendered:
      - rec4: prepare 13.6 min. Pitch: lines-only fit wrong (near
        touchline far off); one tap round fixed the far side (104.1 x
        60.8 m, camera 3.27 m) but not the near touchline (below). The
        fixture's taps were made under the old calibration and sat ~2
        deg off the new panorama (a tilt); moved by matching each tap's
        patch (scores 0.83-0.99). Kits: k=8 put white with red and white
        with yellow, and proposed red against red (purity 0.837); after
        split, 10 groups, merged to white (6) against red (2), the
        yellow referee and a mixed group marked (0.969). Run: 47 min for
        the chunk, ball stage 9.2 min, aim at the ball 60%. Untouched.
      - rec5 (Pixel + iPhone): stops at the pitch step,
        `calibration_failed` (the stitcher can't match the phones: 0/8
        pairs, as when rec5 was first processed; its cameras were set by
        hand). The engine now names this failure instead of "pitch
        failed (exit 1)". With rec5-game's hand-made `cameras.json`
        copied in (the one hand edit), the rest ran: lines-only fit 105.1
        x 64.3 m; with rec5-game's 11 taps 104.4 x 65.4 m, goals at the
        boxes' centres. rec5-game's saved tap fit is 88.3 x 59.3 m from
        the same taps: not explained. Kits: the soft iPhone half blurs
        the crops; split then merge to blue against white, purity 0.772
        then 0.645. A per-phone-model lens table (FINDINGS section 1) is
        what would let rec5 through untouched.
        The joint fit (`jointfit.py`, PR #21) doesn't replace the stitch:
        it starts from phase0/cameras.json. Tested 2026-11-09 from rough
        cameras with no stitcher instead (each phone's tilt and roll from
        the manifest's gravity sensor, focal from `lenses.json`, the two
        phones N deg apart, Main left or right), rec5-game's 11 taps
        (pitch/taps_src.json) and START_POSE at 1.9 / 3 / 8 m, ~10 s a
        fit: from 80-84 deg, Main right, taps 1.0-3.5 px median, 17-30 px
        worst, fitted 80-83.5 deg apart (the hand cameras: 81.0); the fit
        from the hand cameras gives 2.9 / 45.2. The fit keeps the start's
        angle (70 stays 70, worst tap 77-85 px; 90 stays 90, 60-91 px)
        and the wrong side diverges, so a grid over side and angle,
        keeping the lowest tap error, finds both. Pitch from the good
        starts 96-103 x 62-64 m against the hand start's 88.7 x 59.4:
        rec5's yellow paint isn't found in the phones' backgrounds (paint
        median 40 px, the truncation, in every fit, the hand start's
        too), so the fit is the taps' alone and rec5's size stays
        unsettled. Fallback for the engine: when the stitch fails, rough
        cameras (side from the seam's continuity, or ask) for the
        panorama the user taps on, taps required, then the joint fit over
        a grid of angles. Built 2026-11-09 (on top of PR #21): engine
        prepare asks which phone is on the left (the panorama both ways), `jointfit.py
        rough / search / install`, the pitch question's approveAllowed;
        on rec5's taps the search picks 80 deg, fitted 80.8 (hand 81.0),
        taps 2.7 / 19.8 px, in 2 min. End to end on a 4-segment copy of
        rec5 (segments 2-5, untouched): prepare 26 s to the cameras
        question; with "secondary on the left" the rough lines fit 108.9
        x 67.4 m and kits as usual (8.5 min); with rec5's 11 taps (placed
        where they fall on the rough panorama) the search took 88 s and
        picked 80 deg, fitted 80.6, taps 3.2 / 17.5 px, pitch 100.6 x
        64.2 m, 2.67 m up, the lines on the yellow paint; round 2
        approved. The question shows the panorama drawn both ways, not
        two stills: rec5's views cross, so side by side Main's inner edge
        looked like it joined Secondary's (it doesn't), while the wrong
        panorama shows two broken halves. Not yet run through `run`.
- [x] The packaged engine for the Mac app (step D), 2026-11-10, unsigned:
      `packaging/mac/` builds `osb-engine/` (CPython 3.12 from
      python-build-standalone, the 73 pinned packages, an LGPL ffmpeg 8.0.1
      with VideoToolbox and no Homebrew libraries, the code, schemas and
      models, RF-DETR's COCO weights): 1.5 GB, built in under 2 min after
      the first download. accel.py's OSB_PYTHON / OSB_FFMPEG_DIR replace
      `uv run` and the PATH's ffmpeg. As a clean Mac (`test_clean.sh`: empty
      environment, system PATH, a fresh home, model downloads off) on an
      8-segment copy of rec3: prepare 13.7 min to the questions (nothing
      downloaded, nothing written to the home folder), then one chunk of
      `run` in 52 min (ball 15.3 + roles 15.7 min on Core ML, converted on
      first use in ~25 s in all; render 20.4 min), game.mp4 1080p H.264 +
      AAC, 600 s, the intermediates deleted (2.4 GB). Found on the way:
      phase0.py's stitch preview and play.py used x264's `-crf`, which an
      LGPL ffmpeg doesn't have; both use accel.h264_args now.
- [ ] Sign and notarize the engine (needs a Developer ID Application
      certificate; packaging/mac/README.md), then test on a second Mac or a
      macOS VM. coremltools 9 warns it is untested with torch 2.11: the
      exports worked, but pin torch to what coremltools supports if one
      ever fails (detect.py falls back to PyTorch, ~30% slower).
- [ ] The render waits on the ball stage about half of a game (44 vs 24
      min per chunk; roles is 15 of the 44). Run roles in a third worker,
      or alongside the ball stage, so a game takes ~6 h, not 8.5. Check
      the peak memory with all three overlapped (8.4 GB with two).
- [x] Disk: a game needs ~46 GB of output besides its footage at the end
      (rec7: pieces 11 GB, `video.mp4` 12 GB, `game.mp4` 23 GB at 1440p).
      Done 2026-11-09 (engine.py): `run` checks the free space it still
      needs first (remaining pieces + video.mp4 + game.mp4 at 16 / 33
      Mbit/s, ~49 GB for 100 min; `not_enough_disk`), and once game.mp4
      is as long as the chunks it deletes the pieces, video.mp4 and
      audio.m4a (job option keepIntermediates) and writes
      fullgame/done.json so a rerun doesn't render again.
- [x] Verify each segment's SHA-256 against the manifest on download.
      Done 2026-11-09 in the engine (prepare and run; `footage_corrupt`,
      exit 2), each file once (engine/verified.json, by size and
      modification time): 20 GB of rec3 in 6.4 s, the check again 0.3 s.
      The app's download (step E) should check too, to fetch a bad file
      again at once.
- [ ] Android footage: expect ~150 ms missing at every segment switch and
      start times that drift by that much per switch (see the gap-free
      recorder item in the main repo's backlog). Re-sync each segment to the other phone by audio
      cross-correlation rather than trusting `startedAtDeviceMs`, and
      tolerate the undecodable leading frames at the start of segments
      after the first.

## Phase 4-6

- [ ] Ball detection: fine-tune on youth-game footage; Kalman tracking;
      blended camera target.
- [ ] Investigate Roboflow Universe datasets for the ball detector
      (2026-11-09; https://universe.roboflow.com, search "soccer ball",
      "football", "ball"). Community datasets with ball boxes, mostly from
      TV broadcasts, some from fixed or amateur cameras (e.g.
      quince-imaging/soccerballdetector). Licence per dataset (often
      CC BY 4.0, some non-commercial or unstated) and label quality vary.
      Caution: FUT-HEROS found a broadcast-only fine-tune worse than the
      stock model on amateur footage (1/8 goals against 3/8).
      To find out:
      - Which sets are wide, fixed, full-pitch or amateur views like ours
        (ball a few to ~20 px), rather than broadcast close-ups; how many
        ball boxes each has; and whether the licence allows training a
        model we ship (CC BY / CC0 / Apache yes, NC no).
      - Label quality on a sample: missed balls, spare balls, boxes on
        boots or heads.
      - Uses, in order of risk: an extra test set (scored like
        SoccerTrack, never trained on), "not the ball" negatives from
        broadcast frames, then training data mixed with ours. Score any
        retrained detector on every held-out set (rec3 seg9, SoccerTrack),
        with and without the judge.
      - Record each set used in docs/DATA.md (External datasets) with its
        licence, version and what it was used for.
- [ ] Investigate SAM 3 as a teacher that auto-labels balls, to retrain
      RF-DETR on far more and more varied labels (2026-11-09). The
      standard detector is trained on 398 distinct ball boxes from three
      games (rec3 275, rec5 54, rec6 69; footage/models/README.md), few
      for fine-tuning; it still found the ball in 54% of SoccerTrack's
      frames, so the backbone carries and labels are the limit. The judge
      exists mostly to reject ball-sized look-alikes (white boots, line
      crossings, nets: 9 -> 1 wrong on rec3 seg9); enough labelled
      negatives could teach the detector that instead, and the judge
      doesn't carry to new footage (SoccerTrack: 3-4% of real balls kept).
      Prior art: FUT-HEROS
      (https://huggingface.co/build-small-hackathon/futheros-ball-rfdetr)
      tracked the ball with SAM 3 (video, text prompt "ball"), kept frames
      where an independent RF-DETR detection agreed within 35 px, then
      added SAM 3-only frames of the fast, blurred ball in flight; goals
      found on amateur footage 3/8 (stock) -> 7/8 -> 8/8.
      To find out:
      - Licence: SAM 3 is under Meta's SAM License (custom; said to allow
        commercial use with restrictions). Read it before its labels train
        a shipped model.
      - Does it hold our ball? Run it on rec3 seg3 and seg9, rec5 (orange)
        and rec7 (yellow) clips and score its track against the hand
        labels (balleval-style found / wrong) and on SoccerTrack. Speed
        and memory on the Mac (~0.86B parameters, unconfirmed), or a cloud
        GPU.
      - Label rules: agreement with our detector (and the other phone),
        a size check against the pitch map, spot-checks on a sample, and
        what to do with balls in hands, spare balls and balls in flight.
        Negatives: detector candidates SAM 3 rejects where it holds the
        ball elsewhere.
      - Scale: thousands of frames across every game we have (rec2-rec7),
        all colours and light, and later the engine's games. Never from
        held-out time (holdout.py) or SoccerTrack.
      - Retrain RF-DETR on them and compare with today's detector on every
        held-out set (rec3 seg9, new held-out clips) and SoccerTrack,
        with and without the judge. Pairs with the learning curve (25 /
        50 / 100% of today's labels) to see whether label count is still
        the limit, and with upscaled tiles for balls under 8 px, which
        labels won't fix.
- [ ] Investigate SoccerTrack v2 for the follow cam and play analytics
      (2026-11-09; [paper](https://arxiv.org/abs/2508.01802),
      [data](https://huggingface.co/datasets/atomscott/soccertrack-v2),
      [code](https://github.com/AtomScott/SoccerTrack-v2)). 10 full
      university games (934 min) from fixed panoramic cameras (BePro,
      4096 x 1504, 25 fps), the closest public match to our product.
      CC BY 4.0 (data) and MIT (code): commercial use with credit. Per
      frame: every player's and official's pitch position, ID, jersey,
      team and role; 21,428 ball events (12 classes: pass, drive, shot,
      out, throw-in, free kick, goal, ...) with frame and acting player;
      per-game calibration. No measured ball positions (its ball track is
      interpolated between events), so no help for the detector or
      judge. 13.4 GB per game, ~145 GB in all; benchmark split: test
      128057 + 132831, validation 117093 + 132877. Its MMSports 2025
      challenge is player tracking only (Deep-EIoU starter, winner
      GTATrack at HOTA 0.60); no published weights.
      To find out:
      - Follow cam: at each ball event, is the acting player's position
        inside the virtual camera's view? Thousands of scored moments a
        game, against our 26 + 15 hand-drawn must-see boxes. Needs a
        single-panorama input to the pipeline (their panorama and
        calibration in place of our two phones and stitch).
      - Phases of play: out, throw-in, free kick and goal times as a
        test set for detecting restarts and stoppages from footage
        (phones on the mount take no input during a game).
      - Play analytics: player positions and roles to check `roles.py`,
        team sides and our centre of play, and as ground truth for any
        stats we add later (possession, passes, heat maps).
      - Whether players' and officials' positions and the events could
        also train anything (licence allows it), and whether a camera
        this different still predicts how our phones do.
      Start with one test game (128057, 13.4 GB): keep it held out, never
      trained on.
- [ ] The judge doesn't carry to a new camera (2026-11-09, `stbench.py`
      on SoccerTrack, an outside dataset held out for testing: see
      FINDINGS): it keeps 3-4% of real balls the detector found and ranks
      candidates worse than the detector's confidence alone (AP 0.34-0.36
      vs 0.81). Before more per-colour judges: a judge that generalises
      (colour and grayscale augmentation, recoloured white balls, blur and
      scale jitter, fair sampling across games instead of xN copies, or a
      pretrained backbone), scored on every game's held-out labels and on
      SoccerTrack; and check whether follow.py does better trusting the
      detector more where the judge is unsure. Measured since (FINDINGS):
      on rec3's held-out seg9 the judge cuts wrong picks 9 -> 1 (white
      boots, feet, line crossings: ball-sized, so a size check doesn't
      replace it); on rec7's yellow ball judge_v3 finds 0/52 where the
      detector alone finds 38/52 with 1 wrong. Next: the judge as a soft
      weight with a floor, so it can't veto a confident detection, and
      the look-alikes as detector negatives (SAM 3 item below).
- [ ] Held-out ball labels on every game (rule from 2026-11-09, in
      `docs/TESTING-PROCESSING.md`): only rec3's `seg9` is held out;
      rec5, rec6 and rec7 have no independent ball test. Label a held-out
      clip on each (`labelset.py ... clip ... --heldout`; rec6 first:
      daylight). Enforced in code since 2026-11-09 (`holdout.py`: the
      training-data builders leave held-out time out and stamp their
      output; training refuses unstamped or stale data; the engine's
      ball sets hold out the last quarter of each chunk). Every crops.npz
      and balldata/ built before then must be rebuilt before the next
      retrain.
