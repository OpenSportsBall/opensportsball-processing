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
      48, 63, 46, 61% on pieces 3-9. judge_v4 is not benched on rec3,
      rec4cal or rec6 and is not the standard; each new ball colour has
      so far needed its own labels.
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
- [ ] Follow cam speed, further ideas (a full game is ~2.5 h locally
      today, half in each pass):
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

- [ ] Verify each segment's SHA-256 against the manifest on download.
- [ ] Android footage: expect ~150 ms missing at every segment switch and
      start times that drift by that much per switch (see the gap-free
      recorder item in the main repo's backlog). Re-sync each segment to the other phone by audio
      cross-correlation rather than trusting `startedAtDeviceMs`, and
      tolerate the undecodable leading frames at the start of segments
      after the first.

## Phase 4-6

- [ ] Ball detection: fine-tune on youth-game footage; Kalman tracking;
      blended camera target.
