# Updating Nivra

Do the controller first, then the computers that run the Nivra App. Each step says what to check and how to go back.

## Before every update

1. Make a backup: `bin/nivra backup run` on the controller (see [BACKUP.md](BACKUP.md)). The controller also makes
   one every night, but a fresh one takes a minute and is the thing you restore from if something goes wrong.
2. Look at **Studio → Queue**. A running long video is not lost by an update (it continues after the restart), but
   a worker that is mid-task loses that task and the scheduler hands it to another worker. If you can, wait for
   the big jobs to finish.
3. Read the release notes for changes marked "database migration".

## 1. The controller (Raspberry Pi)

```bash
# on your computer, from the Nivra folder (set your own host and user, nothing is stored in the repository)
export NIVRA_PI_HOST=<hostname-or-ip> NIVRA_PI_USER=<user>
bash scripts/deploy-pi.sh
```

`deploy-pi.sh` builds the arm64 binary, copies a package, checks its checksum on the Pi, backs up the old install
(keeping `config/`), unpacks the new one and records the version in `DEPLOYED_VERSION`. It does **not** restart the
controller. Then, on the Pi:

```bash
cd ~/Desktop/Nivra
bin/nivra migrate status      # shows migrations that will be applied
bin/nivra backup run          # backup first
# stop the running controller (Ctrl+C in its terminal, or stop its service), then
bash start.sh controller
```

The controller applies pending **database migrations** when it starts. Migrations are checksummed, run under a
lock, and never rewritten once applied. The log says `applying schema migrations to an existing database` when it
upgrades an existing database. After the start check:

* `bin/nivra doctor` has no FAIL lines;
* Studio opens and **Workers** shows your computers coming back online within a minute;
* **Storage** shows the disk and a recent backup exists (`bin/nivra backup list`).

### Going back

Migrations only add things (tables, columns, indexes, wider role checks). The previous program version keeps
working with a database that was migrated by the next one, except for features it does not know about. If you must
go back anyway:

1. stop the controller;
2. restore the previous program folder (`deploy-pi.sh` kept it under `~/NivraData/backups/`);
3. if the database must go back as well, restore the dump from before the update (see
   [BACKUP.md](BACKUP.md#restore)). Work done since that backup is lost, so prefer fixing forward.

## 2. Computers running the Nivra App (Windows)

Close the app first (**Quit Nivra** in the window or the tray). The installer refuses to update while `Nivra.exe`
is running, before it changes anything.

* Re-run `Nivra-Setup.bat`. When a tagged release is available it downloads the release zip, checks its SHA-256
  against `SHA256SUMS` from the same release, and replaces the program files. It keeps your settings and models
  (`config`, `models`, the Python environment). If no release can be reached it falls back to building from
  source.
* The first start after an update re-checks the hardware (GPU, ffmpeg) and the AI runtime.

The App and the controller speak a versioned protocol (`nivra.worker/1`). Update the controller first and the Apps
after it. There is no in-app "update available" notice yet, so tell your friends when a new release is out.

## 3. Workers on Linux

```bash
cd ~/Nivra && git pull && bash install.sh --role worker --yes     # or unpack the new release
```

## What an update changes in release 0.1 → members and fairness

* **Database migration 0010**: new role `member`, invitations, per-project limits, retry back-off, refund
  counters, and priority bands. Queued video jobs are renumbered into the shared video band.
* **Videos longer than 30 minutes** are refused (the draft of a long video cannot resume yet). Members are limited
  to 10 minutes by default (**Settings → Limits and cleanup**).
* **Nightly backups** start automatically (`backup_enabled`). Install `pg_dump` on the host
  (`sudo apt install postgresql-client`) or keep the `nivra-pg` Docker container running (the controller uses it as a
  fallback).
* **ffmpeg** is required on every computer that makes videos. `Nivra-Setup.bat` installs it. A computer without
  it stops advertising video jobs and says why in the Worker window.
