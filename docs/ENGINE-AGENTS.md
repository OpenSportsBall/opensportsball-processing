# Driving the engine from a script or a model

The Mac app is one way to process a game. `osb-engine` (in a checkout:
`uv run processing/spikes/engine.py`) is the other: everything the app
does with the engine goes through files and commands, so a script, or a
model working for the user, can do the same. This page is written to be
handed to such a model as it is.

The recording folder (the merged `manifest.json` and every segment file)
must already be on disk; the engine downloads nothing and sends nothing.

## The loop

```
osb-engine status    job.json      # one JSON line; "next" is the command to run
osb-engine prepare   job.json      # minutes: sync, pitch fit, kit groups, questions
osb-engine questions job.json      # what is waiting for an answer
osb-engine answer    job.json <question> <flags>
osb-engine prepare   job.json --answers <recording>/engine/answers.json
   ... questions / answer / prepare again until nothing is pending ...
osb-engine run       job.json      # hours: the game video
```

A job is `{"schemaVersion": 1, "recording": "<folder>"}`
(`schemas/engine-job.schema.json`).

- `prepare`, `run` and `status` print progress as JSON lines
  (`schemas/engine-event.schema.json`). `questions` and `answer` print one
  JSON document.
- Exit codes: 0 done, 2 bad job, answers or footage, 3 `run` with
  questions unanswered, 4 a step failed (the `error` event names its
  log), 130 stopped.
- Every command can be run again: finished steps are skipped, and an
  answer to an earlier round is ignored.
- `run` takes hours (rec7: about 8.5 h on an M2 Pro). Start it in the
  background, read its `progress` events (`fraction`, `etaS`), and don't
  start a second one on the same recording. Stopping it (SIGINT or
  SIGTERM) is safe; the same command carries on.

## The questions

`questions` prints, for each pending question: `ask` (what is asked),
`howToAnswer` (the answers and the `answer` flags for each), and the
pictures to look at, as absolute paths. Look at the pictures; don't
answer from the numbers alone.

**cameras** (only when the phones couldn't be stitched): two panoramas,
one with each phone on the left. Say which shows the pitch in one piece.

    osb-engine answer job.json cameras --left main

**pitch**: `overlay` is the empty pitch with the fitted lines in red.
Every keypoint has a close-up (`points.<name>.closeup.image`, 900 x 600):
the fitted lines in red and a ring where the fit puts the point
(`closeup.fit`, in the close-up's pixels).

    osb-engine answer job.json pitch --approve
    osb-engine answer job.json pitch \
        --closeup 'left corner flag=412,318' \
        --closeup 'right 18-yard front corner=455,297' \
        --keep 'halfway far end' --skip 'right corner flag'

`--closeup NAME=X,Y` is the pixel of that point's close-up where the
keypoint really is. Give at least four points spread over both ends;
points not named are left out of the refit. The engine refits and asks
again with the next round, with new close-ups: approve when the lines sit
on the paint. A wrong pitch spoils everything after it, so this is the
question to take time over. When `approveAllowed` is false the points must
be marked once before approving.

**kits**: one contact sheet per group (`groups[].sheet`), each person with
an id under the picture. Every group should be one kit and every kit one
group.

    osb-engine answer job.json kits --ok
    osb-engine answer job.json kits --merge 0,7 --merge 1,2 --keeper 5,6 --not-kit 3,4
    osb-engine answer job.json kits --split 3

`--tapped c0_03,c2_11` names people in the wrong group; it scores the
grouping and moves nobody. A split regroups and asks again, once.

`answer` checks the answer against what was asked (a point or group that
doesn't exist, a pixel outside its close-up, approving when it isn't
allowed) and exits 2 with the reason. It writes only
`<recording>/engine/answers.json`; `prepare --answers` applies it.

## Sharing is the user's decision

The pitch and kit answers carry a `share` switch: whether the user agrees
to share that result (pictures of the pitch and of people) to improve the
models. The app shows it to the user. `answer` records `share: false`
unless `--share` is given. A model must not pass `--share`, or run
`osb-engine share` or `label`, unless the user has said so for this game.

## What a model should hand back to the user

- The pitch overlay after the last round, and the kit sheets, if the user
  wants to check them: these two answers decide the video's quality.
- From the `done` event: `output` (the video), `ballAimShare` and
  `ballLow`. `ballLow: true` means the models don't know this ball: the
  video follows the players more than the ball.
- Don't wait for `done` to say so: `run` prints a `ball` event after each
  chunk the camera has followed, and from the second chunk it carries
  `ballLow` (one chunk can't tell: rec7 gave 31% for its first and 0-2%
  for the rest). `status` has the same fields at any time, and
  `chunkDetails`: each chunk's two stages, how long they took, their logs
  and the chunk's ball share.
- When `ballLow` is true, tell the user, and that ball labels teach later
  models (not this video). `label <job> ball` works while `run` is still
  going, from the chunks done so far; it is the user's decision, as above.
- On exit 4: the `error` event's `message` and `log`.
