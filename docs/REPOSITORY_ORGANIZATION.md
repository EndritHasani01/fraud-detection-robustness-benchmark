# Repository organization record

Organized on 9 September 2026 for the course-project handoff.

## Preservation policy

No original file was intentionally deleted. Superseded notebook versions, historical v3 exports, duplicate archive ZIPs, development plans, troubleshooting reports, CV notes, and report design assets were relocated to `_local_archive/`, which Git ignores. Original files edited for navigation were backed up there before editing. The move ledger `_local_archive/move-manifest.csv` records original paths, destination paths, and SHA-256 hashes. Markdown link repairs have their own original backups under `before-link-repair/`.

The final PDF, Word report, and PowerPoint were selected from the former `final-report/report/` directory and moved into `final-report/` unchanged. The older root Word export was open in another application: its archive copy was preserved and the original was left in place and excluded from Git.

## Retained structure

The final evidence bundle retains its original directory name so report references and evidence paths remain recognizable. The clean notebook stays at the root to preserve its test/synchronization contract. The executed output notebook is under `notebooks/`; technical background is under `docs/`. All frozen configurations and vendored research repositories remain available for compatibility and provenance.

## Git behavior

An ignored archive stays on this computer and is not included in future commits. Moving an already tracked file into it appears in Git as removal of the old tracked path; the bytes remain in the local archive and historical commits. This is expected and is distinct from deleting local files. The reorganization is recorded in Git on user request. No push, history rewrite, or email is performed.

Historical prose can refer to archived experiments or old runtime paths. Such references are contextual history, not required inputs for the final handoff. Start from the root README and evidence index for the current reading path. Empty source directories left by file moves are also preserved locally rather than deleted.

## Validation at handoff

- 551 relocation-ledger entries verified against SHA-256 hashes; Markdown originals are preserved before link edits. A reviewable copy is [RELOCATION_MANIFEST.csv](RELOCATION_MANIFEST.csv).
- 79 local links in the landing page, technical report, and primary handoff guides resolve.
- `py -m unittest discover -s tests`: 121 tests passed (15.954 seconds).
- `py tools/sync_kaggle_notebook.py --check`: notebook modules synchronized.
- `git diff --check`: passed.
- No working-tree changes to benchmark source, frozen configs, regression tests, synchronization tool, or the clean notebook.
- Retained `results.csv`: 5,040 rows, all `status=ok`, evenly divided between both protocols.

The older generated `runs/` directory is preserved at `_local_archive/runs/`; new executions recreate ignored `runs/`. The original Word export remains locally because it was locked, with a verified archive copy. The handoff commit records the reorganized files and untracks archived originals, including that older root export. Archived files are not available to fresh clones, so keep a separate backup of the archive if its historical contents are needed elsewhere.
