# Task: build the master tables `images.csv` and `boxes.csv`

You are helping a student group (AAU, P3) build a drone detector that must signal its own uncertainty under domain shift. The detector is trained on ground-view RGB data and tested on held-out domains. Your job is to turn many raw drone datasets, each with its own annotation format, into two clean master tables with a leakage-free train/val/test split. The tables feed both the EDA and the later YOLO/DETR export.

**If anything is unclear, ask me instead of guessing.** Several things below are explicitly marked "ask me".

Work in phases. **Stop after each phase**, show me what you found, and wait for my approval before continuing.

---

## Ground rules

- **Raw data is read-only.** Never modify, move, rename or delete anything in the dataset folders, on my Mac or on the external hard disk.
- Write all outputs inside the AILAB folder: tables in `master/`, scripts in `scripts/`, logs and reports in `master/logs/`.
- Put all logic in scripts (Python, pandas), so everything can be re-run from scratch. No manual edits to the CSVs.
- Never assume an annotation format. Open real sample files first and show them to me.
- Ask before anything that takes more than ~30 minutes or writes more than ~10 GB (e.g. extracting video frames).
- Use multiprocessing for slow per-image steps (reading image sizes, hashing), and cache intermediate results so a crash doesn't mean starting over.

---

## Phase 0 – Find the data (read only, write nothing)

1. Read `CLAUDE.md` in this folder if it exists. Where it conflicts with this prompt, this prompt wins.
2. The raw data is on an **external hard disk** as well as in the AILAB folder. List `/Volumes/` and ask me which volume to use. If you get "Operation not permitted", tell me – I need to grant the app access to removable volumes in macOS settings.
3. Make an inventory of every dataset you find, on both the disk and in AILAB. For each dataset, report:
   - Folder structure (top 3–4 levels) and file counts by extension
   - Images or videos? If videos: are frames already extracted anywhere?
   - Annotation format, with **one real sample file shown** per dataset
   - Class names that occur in the annotations
   - How the official splits are defined (folders, list files, none)
   - Any existing `*_meta.csv` file for that dataset
4. Report which of these candidate datasets you found, and anything else that looks like a drone dataset:
   DrIFT, DUT Anti-UAV, Seraphim, WOSDETC Drone-vs-Bird, Anti-UAV300, Anti-UAV410, CST Anti-UAV, Det-Fly, MAV-VID, TIB-Net, Drone-detection-dataset (Svanström et al., Halmstad).
5. **Ask me** which datasets to include before Phase 1.

---

## Phase 1 – One converter per dataset

Write one converter script per dataset, reading the raw annotations into a common in-memory format. Convert **one dataset at a time** and validate it before starting the next.

### Box format traps (verify each on real data)
- **Anti-UAV300 / Anti-UAV410:** per-sequence JSON with `exist` (1/0) and `gt_rect`. The Anti-UAV paper describes `gt_rect` as [x1, y1, x2, y2], but the files may use [x, y, w, h]. Test it: if the third value is often smaller than the first, it is a width. Show me the evidence before deciding. Frames with `exist = 0` are negatives.
- **DUT Anti-UAV:** probably Pascal VOC XML (corner coordinates) – verify.
- **YOLO-format datasets** (e.g. Seraphim, maybe others): normalized `class cx cy w h` – convert back to pixels using the real image size.
- **COCO JSON datasets:** `[x, y, w, h]` from the top-left corner, in pixels.

### Frames from videos
- Use **every 10th frame** of each video. If frames were already extracted (check the `*_meta.csv` files and existing folders), reuse them and make sure frame numbers line up with the annotation indices.
- If frames must be extracted from video, ask me before writing them, and tell me the expected disk use.

### Validation per dataset (show me the results)
- Box counts, image counts and empty-frame shares per split match the existing `*_meta.csv` file, if one exists. Explain every mismatch.
- Draw boxes on ~10 random images per dataset into `master/logs/samples/<dataset>/` so I can check the boxes visually.

---

## Phase 2 – Build `images.csv` (one row per image)

Use exactly these columns. Use the literal value `unknown` for unknown categorical values; never leave a categorical cell empty.

| Column | Content |
|---|---|
| `image_id` | Unique key: `<dataset>/<sequence_id>/<modality>/<frame>`, or `<dataset>/<file stem>` for datasets without sequences |
| `dataset` | `anti_uav300`, `anti_uav410`, `cst`, `dut`, `drift`, `seraphim`, `det_fly`, `drone_vs_bird`, `mav_vid`, `tib_net`, `svanstrom` (only the ones included) |
| `image_path` | Absolute path to the image file (on the disk or in AILAB) |
| `sequence_id` | Video/sequence name. DrIFT: filename prefix before the last `_` (the capture timestamp). DUT and Seraphim: = `image_id` |
| `source_sequence_id` | The group key for splitting. For Anti-UAV410 sequences whose name also exists in Anti-UAV300: the Anti-UAV300 name. Otherwise = `sequence_id` |
| `frame_idx` | Frame number within the video; empty for still-image datasets |
| `orig_split` | The dataset's own split (`train`/`val`/`test`), or `none` |
| `new_split` | Our leakage-free split (Phase 4) |
| `img_phash` | Perceptual hash (`imagehash.phash`, as hex string) |
| `width`, `height` | Native resolution in pixels, read from the image file itself |
| `n_boxes` | Number of **drone** boxes |
| `is_negative` | `True` if `n_boxes == 0` |
| `sensor` | `rgb` or `ir` |
| `source` | `real` or `synthetic` |
| `view` | `ground`, `aerial` or `unknown` – see rules below |
| `background_raw` | The dataset's own background value (CST, DrIFT); otherwise `unknown` |
| `background` | `sky`, `vegetation`, `urban`, `terrain`, `water`, `mixed` or `unknown` – see mapping below |
| `weather` | `normal`, `foggy`, `rainy`, `snowy` or `unknown` (DrIFT only) |
| `season` | `summer`, `fall`, `winter`, `adverse` or `unknown` (DrIFT only; empty DrIFT season = `adverse`, because season is only recorded for normal weather) |
| `camera` | `infiniti`, `ptz` or `unknown` (DrIFT only) |
| `illumination` | Anti-UAV300: `low` if `LI` is among the sequence's attributes, else `normal`. All others: `unknown` |
| `attributes_raw` | The raw challenge-attribute string (TC, FM, …), unchanged. Not decoded or analysed |

### `view` rules
- DrIFT: from its own metadata.
- Det-Fly: `aerial` (air-to-air, recorded from another UAV).
- TIB-Net, Drone-detection-dataset (Svanström): `ground`.
- Anti-UAV300, Anti-UAV410, CST, DUT: no official viewpoint annotation. Look at ~20 random images per dataset, report what you see, and **ask me** before labelling them `ground`.
- Seraphim, MAV-VID, Drone-vs-Bird and anything else: `unknown` unless I say otherwise.

### `background` mapping
- CST: building → `urban`, urban-areas → `urban`, jungle → `vegetation`, cn_mountains → `terrain`, water → `water`, cn_sky → `sky`.
- DrIFT: `0` → `sky`, `0;1;2` → `mixed`. Codes `1` and `2` are tree and ground, but the order is **unconfirmed**. Save ~10 example images for each of code 1 and code 2 to `master/logs/drift_background_check/` and **ask me** to confirm before mapping them to `vegetation` / `terrain`.
- All others: `unknown`.

---

## Phase 3 – Build `boxes.csv` (one row per drone box)

**Only drone boxes are kept.** Labels like UAV, drone, Drone, multirotor, fixed-wing and similar all count as drone; show me the full list of raw class names and how you mapped them. Boxes of other classes (bird, airplane, helicopter, …) are **not** written to `boxes.csv` – log them to `master/logs/dropped_boxes.csv` with dataset, image_id and class name. An image whose only boxes are non-drones becomes a negative (`n_boxes = 0`).

| Column | Content |
|---|---|
| `box_id` | `<image_id>_<index>`, index from 0 |
| `image_id` | Link to `images.csv` |
| `class_raw` | The original label as written in the annotation |
| `class_id` | Always `0` |
| `x_min`, `y_min`, `x_max`, `y_max` | Master coordinates in original pixels (floats), clipped to the image |
| `box_w_px`, `box_h_px` | `x_max - x_min`, `y_max - y_min` |
| `box_min_side_px` | `min(box_w_px, box_h_px)` – **the measure used for both size classes** |
| `sqrt_area_px` | `sqrt(box_w_px * box_h_px)` |
| `cx_norm`, `cy_norm`, `w_norm`, `h_norm` | Center and size normalized by image width/height (0–1) |
| `rel_area` | `box_w_px * box_h_px / (width * height)` |
| `aspect` | `box_w_px / box_h_px` |
| `size_class` | From `box_min_side_px`, COCO-style thresholds: `small` < 32, `medium` 32–96, `large` > 96 |
| `size_class_svanstrom` | From `box_min_side_px`: `detection` < 5, `recognition` 5–15, `identification` > 15 |
| `source_format` | `antiuav_json`, `voc_xml`, `coco_json`, `yolo_txt`, or another short name for the format it came from |
| `was_clipped` | `True` if the original box extended outside the image and was clipped |

Drop boxes with zero or negative width/height after clipping, and log them in `master/logs/invalid_boxes.csv`.

Do **not** add `target_size`, `pad_x`, `pad_y`, `crop_x` or `crop_y`. Those belong to a later export step, not to the master tables.

---

## Phase 4 – Duplicates and leakage-free split

### 4a. Duplicates
- Find near-duplicates across **all** datasets with `img_phash` (Hamming distance ≤ 5 as a starting point; show me a few example pairs around the threshold so we can tune it).
- Seraphim is curated from 23 other datasets and may contain images from the others. Report how many Seraphim images match images from each other dataset.
- Anti-UAV300 and Anti-UAV410: 310 Anti-UAV410 sequences have the same names as Anti-UAV300 sequences (the IR halves). Confirm this, and check whether their frames are pixel-identical or near-identical. Don't count the same frame twice: **ask me** which copy to keep.
- Write all findings to `master/logs/duplicates.csv`.

### 4b. Split groups
Build split groups so that nothing that looks alike can land in two different splits:
1. Start from `source_sequence_id`.
2. Merge groups that contain near-duplicate images of each other (union-find over the duplicate pairs from 4a).
3. Every group goes into exactly **one** split.

### 4c. Split rules
- Keep each dataset's official **test** set as test where it is already clean (no group shared with train/val). This keeps our numbers comparable with published results.
- **DrIFT:** 31 of its 62 captures are split frame-by-frame between train and val in the official split. Re-split train/val by capture. Keep the official test captures as test.
- **CST:** keep the official split, but note in the report that, per the paper, train and val are non-overlapping clips of shared source videos.
- **Anti-UAV300/410:** 8 of the shared sequences sit in different splits in the two datasets. Give each source sequence one split and tell me what you chose and why.
- Datasets without an official split: split by group, roughly 70/15/15 by image count, with a fixed random seed.
- **Held-out domains:** the project needs at least one domain that is never trained on (candidates in the report: Det-Fly / aerial view, DrIFT adverse weather, IR). **Ask me** which domains to hold out before finalizing. Held-out images get `new_split = heldout`.

---

## Phase 5 – Final checks and report

Write `master/logs/build_report.md` containing:
- Rows per dataset and per `new_split`, and box counts per dataset
- All checks below, each marked pass/fail:
  - `n_boxes` in `images.csv` equals the number of rows per `image_id` in `boxes.csv`
  - Every box satisfies `0 ≤ x_min < x_max ≤ width` and `0 ≤ y_min < y_max ≤ height`
  - Every `image_path` exists
  - No split group appears in more than one `new_split`
  - No near-duplicate pair crosses splits
  - No empty categorical cells (only real values or `unknown`)
- Every decision you made, with the reason, and every open question

Finally, show me the first 5 rows of each table and the report summary.
