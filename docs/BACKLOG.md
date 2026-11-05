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
        clean). Clock drift is a steady -9 ppm (~50 ms per game, ~2.5 ms
        per 5-minute segment), so one offset per segment is enough.
      - Re-sync every segment by audio. The offset jumps by up to 390 ms
        at segment switches (the phones lose different amounts), and
        `startedAtDeviceMs` doesn't show it: BLE+metadata was 10 ms off
        for segment 0 and 309 ms off by segment 20.
      - Stitch: OpenCV's panorama stitcher (estimated focal lengths and
        rotations, graph-cut seam, multi-band blend). One calibration from
        segment 0 held for the whole mounted game: overlap points moved
        ~3 px median at 960 px wide in every segment, the same as the
        measurement noise within one segment. The handheld clip drifted
        (45-57 deg apart), so the pipeline needs a rigid mount or a
        periodic re-estimate. Single-pair estimates of the angle are
        noisy at night (78.8-86.8 deg); compare homographies instead.
      - A homography fitted on field points did worse than the stitcher:
        turf texture gives mostly false matches.
      - 97.5 min of 4K on battery: thermal nominal throughout; battery
        100 to 74% (Pixel 11) and 55% (11 Pro); ~24 GB per phone.
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
      both lenses need to be known, not one. Plan:
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
      M-series Mac. Still to do: cross segment boundaries.
- [ ] Follow cam aim, judged by eye on 2 minutes of the night game
      (segment 9, 60-180 s; 2026-10-15). Best so far: "A",
      `follow.py ... play goalkick,goalframe,unsure` (hotspot + goal kicks
      + goal in frame + zoom out when unsure): the only version that
      catches every key moment in the clip, but zoomed out at times
      (median 55 deg). Runner-up: `play goalframe`. All tuned on this one
      clip; check another stretch of the game. What each step fixed:
      - Speed-weighted players beat the median of all players (caught a
        quick counter; in play.py's test, ball in a 60 deg view 100% vs
        66%), but an average of 3-6 runners jumps between groups.
      - Hotspot of running (pooled +-0.5 s, weighted by speed, sticky:
        another spot must be 50% stronger for 0.7 s): followed the attack
        to the left goal; aim jumps > 40 deg/s dropped 435 -> 73 frames.
        Needed linear (not squared) speed weights and 0.5 s position
        smoothing: far-side jitter read as running.
      - Aim only from people inside the lines; spots by a touchline
        occupied >= 10% of the time (bench, subs) are ignored.
      - Lull hold: <= 1 runner for 1 s keeps the aim from just before.
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
      the static filter at 30 s, a 2-hit minimum, judge >= 0.3 and a
      0.6 s gap: 54% / 62% (wrong 1 / 8; with a looser tolerance 57% /
      66%, wrong 0 / 4). Now the defaults. Of the remaining misses about
      half have no detector candidate near the ball at all, the rest are
      balls on the near touchline just outside the pitch margin, in a
      player's hands (the judge scores those ~0), or in tracks the chain
      didn't pick. Also fixed: ballclip.py only read the first segment
      file, so seg 3's candidates stopped at 1:00; with the full clip
      A + ball + goalin captures 10/13 play boxes.
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
      judged ball, path 7% of rec5 0-10 min). 98 labels on rec5's Pixel
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
      lost 4 found (46 -> 42/53): watch white-ball games.
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
      minimum lull length). The ball-stage changes mean a full re-render
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
      - The starting pose is hand-read from the panorama. Automate it
        (centre circle ellipse + halfway line give distance and height),
        with a tap-two-points fallback in the app (centre spot, a corner).
      - Assumes a regulation layout and full-size goals; youth and
        small-sided pitches need a pitch type or more free sizes.
      - Thresholds tuned on one game; test other venues and daylight.
      - Goals moved mid-game (portable) would need a per-segment re-check.
      Regression test: `processing/tests/test_pitch_regression.py` reruns
      the whole mapping on saved background panoramas and checks goals,
      size, camera and 15 key points (<= 0.1 deg) against results checked
      by eye. Fixtures hold footage, so they stay out of git
      (`footage/fixtures/pitch/<name>`, made by `pitch.py <rec> fixture
      <dir>`; back them up privately). One so far: the night game.
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
        label a second game before training more.
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
