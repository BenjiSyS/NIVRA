# Backup and restore

## What matters

| Thing | Where | Why it matters | Backed up automatically |
|-------|-------|----------------|-------------------------|
| PostgreSQL database | Pi: container `nivra-pg`, data in `~/NivraData/postgres` | users, projects, devices, jobs, models, datasets, runs | yes (`database.dump`) |
| Secrets | `<data_dir>/secrets/` (`controller-secrets.json`, `database.env`) | without it project codes and device identities stop working and passwords cannot be verified | yes (`secrets.tar.gz`) |
| Artifacts (videos, model weights, datasets) | `<data_dir>/artifacts/` | big, mostly re-creatable (retrain, regenerate) | **no**, copy it yourself if you care |
| Configuration | `config/controller.json` | small | no, copy it |

A backup on the same disk does not survive a dead disk. Copy `~/NivraData/controller/backups/` to another
device now and then (USB drive, another computer).

## Nightly backup (automatic)

The controller makes one backup per day, after 03:00 local time (shortly after a start if one is overdue), into
`<data_dir>/backups/backup-YYYYMMDD-HHMMSS/`, and keeps the newest 7.

Settings in `config/controller.json`:

| Key | Default | Meaning |
|-----|---------|---------|
| `backup_enabled` | `true` | **Off switch.** `false` (or env `NIVRA_BACKUP_ENABLED=0`) disables the nightly backup |
| `backup_retain` | `7` | how many backups to keep (1 to 365) |
| `backup_hour` | `3` | local hour after which the daily backup runs |
| `backup_dir` | `<data_dir>/backups` | for example a mounted USB disk |
| `pg_dump_path` | search `PATH` | full path of `pg_dump` |
| `backup_docker_container` | `nivra-pg` | when the host has no `pg_dump`, run it inside this running container |

The database password is passed to `pg_dump` through its environment, never on a command line. Backup files and
folders are readable by the owner only.

Needs one of: `pg_dump` on the host (`sudo apt install postgresql-client`; use a version at least as new as the
server), or Docker with the `nivra-pg` container running. If neither is there the controller logs
`nightly backup failed` with the reason and retries hourly. Look for it in **Studio → Logs** (service
`controller`).

## Back up now

```bash
bin/nivra backup run        # same as the nightly one
bin/nivra backup list
```

## Restore

Practise this once on a spare database before you need it.

```bash
# 1. stop the controller
# 2. restore the secrets if the secrets folder is lost
tar -xzf backup-.../secrets.tar.gz -C ~/NivraData/controller/        # creates secrets/
# 3. restore the database (custom format) into an empty database
createdb -h 127.0.0.1 -p 5433 -U nivra nivra_restored
pg_restore -h 127.0.0.1 -p 5433 -U nivra -d nivra_restored --no-owner backup-.../database.dump
# 4. point the controller at it (database_url in secrets/database.env), start it, run bin/nivra doctor
```

With the Docker container: `docker exec -i nivra-pg pg_restore -U nivra -d nivra_restored --no-owner < database.dump`.

To test a backup without touching production, restore into a database whose name ends in `_test` or on another
computer and open Studio against it.

If you lose the `secrets` folder and have no copy: users can still be recreated (`nivra admin create`), but every
enrolled computer must enroll again (new project code) and API tokens must be created again.

## Rotating passwords and the project code

* Admin password: `bin/nivra admin passwd --username admin` (signs that user out everywhere).
* A friend's password: same command with their username, or turn their account off in **Studio → People**.
* Project code: Studio → top right → Manage → Rotate. Enrolled computers keep working; only new enrollments need the
  new code.
