# Stanford Drone Dataset (annotations only)

| | |
|---|---|
| **Role** | Route-choice validation and academic demonstration only: real trajectories of pedestrians, cyclists and others across a university campus, including off-path routes. |
| **Provider** | Robicquet et al., "Learning Social Etiquette", ECCV 2016 (Stanford CVGL). The official download no longer works, so the files come from the Kaggle mirror `aryashah2k/stanford-drone-dataset`, which is complete. |
| **License** | **CC BY-NC-SA 3.0**, the original's. The mirror's own label is wrong and can't relicense it. No model fitted to this data may appear in shipped or distributed artifacts (CLAUDE.md core rule 5); nothing under `src/features/` reads it, and a test checks that. Cite Robicquet et al. 2016. |
| **Access** | `python -m src.ingest.stanford_drone` (no Kaggle login needed for a public dataset), then `--parse`. `--with-reference` also fetches each video's one reference still (about 10 MB in all). Videos are never fetched. |
| **Files** | `data/raw/stanford_drone/annotations/<scene>/<video>/annotations.txt`, or `.txt.zip` where Kaggle served it zipped (the larger files), with a `MANIFEST.json` per video folder. About 72 MB as held, 445 MB unzipped. |

## What the parse gives (25 October 2026)

| Table | One row per | Contents |
|---|---|---|
| `tracks.parquet` | object and frame (10,616,256 rows) | `scene`, `video`, `track_id`, `frame`, box centre `x`, `y` in pixels of the scene's reference frame, `label`, `lost`, `occluded`, `generated` |
| `window_counts.parquet` | video, label and 1,800-frame window (974 rows) | `in_view`: objects seen in the window; `first_seen`: objects seen for the first time |

- 8 scenes and 60 videos: bookstore 7, coupa 4, deathCircle 5, gates 9, hyang 15, little 4, nexus 12, quad 4.
- 10,300 tracks: 5,232 pedestrians, 4,210 cyclists, 316 cars, 292 skaters, 174 carts and 76 buses.
- Videos run from 430 to 14,557 frames (median 11,007).
- A median of 19 pedestrians are in view per window, and at most 116.

## Known issues

- **Most positions are interpolated.** 98.4% of rows have `generated` set: annotators marked key frames and the tool filled the rest. Positions between key frames are straight-line guesses, so use them for routes, not for speeds.
- **`lost` rows are off-screen** (37.4% of rows). The counts skip them.
- **Pixels, not metres.** There's no published scale per scene. Comparing route lengths across scenes needs a scale estimated from the reference stills (`--with-reference`) and a known length in each scene.
- **The windows are counted in frames.** 1,800 frames is a minute at the videos' 30 frames per second (`stanford_drone.frames_per_window` in `configs/data.yaml`).
