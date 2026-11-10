# schemas

Pinned copies of schemas whose source of truth is `shared/schemas/` in the
main OpenSportsBall repository (Apache-2.0). Don't edit the copies here;
change them there and re-sync.

- `manifest.schema.json`: the upload manifest, also used by the phone app
  and the API.
- `engine-job.schema.json`, `engine-questions.schema.json`,
  `engine-answers.schema.json`, `engine-event.schema.json`,
  `engine-labelset.schema.json`, `engine-label-answers.schema.json`,
  `engine-contribution.schema.json`: the processing engine's files
  (`processing/spikes/engine.py`), which the Mac app and the cloud job
  read and write.

Pinned from main repo commit: `7d113c4`

Check or update the copy against a local checkout of the main repo:

```
python3 scripts/sync_schema.py ../SportsBallVideo          # check
python3 scripts/sync_schema.py ../SportsBallVideo --update # copy it over
```

Every manifest carries `schemaVersion`. Processing should reject a
version it doesn't know rather than guess.
