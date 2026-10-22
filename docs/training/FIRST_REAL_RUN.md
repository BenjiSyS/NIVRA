# Checklist: first real training run

The commands live in [TRAINING.md](TRAINING.md#train-on-your-own-videos), [CAPTIONING.md](CAPTIONING.md) and
[DATA_PIPELINE.md](DATA_PIPELINE.md). This page only puts them in order and says what to check between steps.
Nothing here has been run on real video yet: durations and disk use are **not** stated because they have not been
measured. Measure on the first batch and write them down.

## Before you start

- [ ] Backups work: `bin/nivra backup run` on the controller succeeds, and a copy is off the Pi ([BACKUP.md](../installation/BACKUP.md)).
- [ ] Migration 10 is applied and Studio → **Storage** shows plenty of free disk. The server refuses uploads when less than
      10 GB (or twice the upload) is free; plan for the raw clips, the encoded latents and the checkpoints.
- [ ] The training computer has the AI runtime and ffmpeg (Worker window: no notice about missing ffmpeg; `nivra doctor` has no FAIL).
- [ ] The computers that will train are marked **trusted** in Studio → Workers (HD training and encoding jobs normally require it).
- [ ] Friends' limits are set (Studio → Settings → Limits and cleanup). Training keeps a reserved share of the helper
      computers even while friends generate videos, but a long video flood still slows it down.
- [ ] You own, or have the right to use, every video you train on. The repository ships no data.

## Steps

1. **Collect** a folder of short clips of one look (see the style notes in DATA_PIPELINE.md).
2. **Caption**: `nivra caption <folder> --limit 100`, repeat until nothing is left; open a few `.txt` files and fix obvious errors.
3. **One-machine smoke test**: `python -m nivra.cli train-pipeline --config nivra_hd_tiny --videos <folder> --out <run>`
   finishes quickly and proves the data is readable.
4. **Real pipeline** with the real config on the GPU computer, same command with `--config nivra_base` (GPU memory use is an
   estimate until it has run on the RTX 2060).
5. **Push the dataset** to the controller: `nivra dataset push <run>\latents ...` (see TRAINING.md). Check Studio →
   Datasets shows the shards as complete and Storage shows the new size.
6. **Start the distributed HD run** in Studio → Training (or the API), with at least two trusted workers if you have them.
7. **Watch** Studio → Training (loss curves, per-worker updates) and Logs. The evaluation gate decides whether a round's
   model is promoted; a model that fails the gate is not published.
8. **Promote** the winning checkpoint, import or activate it as a model version, then generate a few clips from the App.
9. **Clean up**: Storage → delete temporary files, old failed videos; keep the dataset and the best checkpoints (checkpoint
   retention never deletes protected, initial, best or latest ones).

## Stop conditions

Stop and investigate when: loss becomes NaN, a worker keeps failing the same job (Studio → Queue shows the reason), disk
free drops below 20 GB (Storage warns), or the Pi's database shows errors in Logs. The run can be paused and resumed;
it survives controller restarts.
