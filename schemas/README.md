# schemas

`manifest.schema.json` is a **pinned copy** of the upload manifest schema.
The source of truth is `shared/schemas/manifest.schema.json` in the main
OpenSportsBall repository (Apache-2.0), which the phone app and the API
also use. Don't edit the copy here; change it there and re-sync.

Pinned from main repo commit: `7d113c4`

Check or update the copy against a local checkout of the main repo:

```
python3 scripts/sync_schema.py ../SportsBallVideo          # check
python3 scripts/sync_schema.py ../SportsBallVideo --update # copy it over
```

Every manifest carries `schemaVersion`. Processing should reject a
version it doesn't know rather than guess.
