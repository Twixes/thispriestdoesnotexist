# Resource cleanup fix

Scope: only `diagnostic.py` implementation changed. Its one `pins.json` file entry was refreshed; all other 401 file pins, all 47 package versions, and the Python version are unchanged. Original script, pins and readiness evidence are byte-preserved in `pre-resource-fix/` with a SHA256 manifest. `resource-fix.patch` is the exact implementation diff.

- Worker verifies final peak RSS is below 6 GiB and final wall-clock time is before the original deadline after final input/pin checks and before writing successful result.
- Supervisor independently rejects excessive or invalid worker-reported peak, excessive sampled peak, nonzero worker exit, incomplete result, and final elapsed time at or beyond 600 seconds. It records the same checked final elapsed value.
- All supervisor exits after worker creation clean its owned session/process group with SIGKILL and wait for the leader, including already-exited leaders whose descendants may remain. A missing group is tolerated. Cleanup errors cannot report success. SIGTERM raises a catchable exit; KeyboardInterrupt/SystemExit and other exceptions are recorded as failures. SIGINT/SIGTERM are ignored only during cleanup and result writing, then restored.

Validation: 8 existing stdlib static tests passed; 18 additional mocked cases passed, including final-only RSS/deadline crossings, monitoring limit failures, interrupted/failed spawning or monitoring, malformed worker results, nonzero worker exit despite a success file, cleanup failure, and a missing owned group. The worker final guard statements were exercised separately with stdlib mocks. No child process was created by these fixtures, no ML library imported, and no model loaded or diagnostic executed. All 402 file pins, 47 package versions and 32 reference input hashes verified. This does not claim an actual operating-system signal integration test or runtime restoration success.

New diagnostic SHA256: `15772fd69e92467bf7785cb4b955307dfeb14f55ec954e88b2fb7bc30cb5d1b5`

New pins SHA256: `1e0b5ac3c455ea56c288dd83e7314e0a319ceda12b87d9fcceb322e339533d99`

Reproducible checks: `research/restoration/.venv/bin/python < research/restoration/resource-fix-tests.txt` and `research/restoration/.venv/bin/python research/restoration/test_static.py`. Evidence: `resource-fix-checks.json`, `static-tests.log`, `static-readiness.json`, and `static-evidence.json`.
