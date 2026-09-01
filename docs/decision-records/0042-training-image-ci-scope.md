# 0042. Separate training-image acceptance from repository-wide smoke tests

- Status: Accepted
- Date: 2026-09-01
- Owners: perception training/research 담당자
- Supersedes: none

## Context

The repository-wide smoke job runs every Python test, including ROS autonomy safety
contracts. After commit `b2b3a4c3604dfd7faede4b68f0f10e01041ed1f7` changed the deployed
planner/controller speed configuration to `0.10..1.00 m/s`, the preserved autonomous
speed test and decision record 0029 still required `0.30..3.00 m/s`. That mismatch is
real and must remain visible until the autonomy owners resolve and record it, but it
does not affect the immutable MMSeg training image, B5 provenance gate, Semantic20
data contracts, or model training code.

The Docker workflow previously reused the entire repository-wide suite as its image
acceptance gate. As a result, an unrelated ROS configuration mismatch prevented an
otherwise unchanged training image from being tagged by Git SHA and pushed for the
pre-registered B5 run.

## Decision

Keep `.github/workflows/code-smoke.yml` unchanged as the repository-wide test job.
It continues to run `python -m unittest discover -s tests -v`, including autonomy and
ROS tests, so the speed-contract failure is neither skipped nor hidden.

Change `.github/workflows/docker-build.yml` to use an explicit allowlist of training
image contract modules. The allowlist covers data conversion, B2/B5 provenance and
resolved configs, MMSeg integration, evaluation, runtime contracts, Semantic20
training/evaluation/handoff, TA0, and target adaptation. The Docker job also runs the
repository artifact guard before building and still verifies the immutable
`ADOM_GIT_SHA` and packaged Semantic20 resources before pushing.

No test method receives a skip marker and no autonomy expectation or runtime YAML is
changed by this decision.

## Rationale and evidence

The failed assertion reads the planner and local-controller YAML files and compares
them with the accepted autonomous speed profile. It does not import or exercise the
B5 configs, checkpoint provenance validator, Semantic20 datasets, MMSeg training
loop, GPU doctor, or image source lock. PR 51's B5-specific tests passed in the same
run before the independent autonomy assertion failed.

An explicit positive allowlist is preferable to ignoring a named failing test: the
Docker job documents exactly which repository surfaces authorize a training image,
while the full smoke job continues to detect both the current mismatch and future
cross-repository regressions.

## Alternatives considered

- Add `skip` to the failing autonomy test: rejected because it would hide an
  unresolved safety-contract mismatch.
- Change the autonomy expected value or YAML in the B5 PR: rejected because the
  correct operational speed profile belongs to the autonomy owners and requires a
  separate decision.
- Push an untested image manually: rejected because it would bypass the immutable
  image and training contract gates.
- Wait for the autonomy change before B5: rejected because the two validation
  surfaces are independent and can remain visibly separated.

## Consequences

- A Git-SHA training image can be published when every training-image contract test,
  the artifact guard, and the immutable image checks pass, even if repository-wide
  autonomy smoke remains red.
- PR and main branch status still expose the autonomy failure until it is fixed.
- Adding a new training-critical test requires adding its module to the Docker
  allowlist as part of the same change.
- This decision does not authorize B5 training by itself; GO evidence, exact GPU
  doctor, dataset lock, memory probe, and canonical-test lock remain mandatory.

## Validation and rollback

Validate the allowlist inside the built dependency image and confirm the image push
step runs only after it passes. Run `python scripts/check_git_artifacts.py` locally
and in the Docker workflow. If a training path begins depending on an omitted module,
add that module explicitly or restore the full suite after the autonomy contract is
resolved; do not silently add a broad test exclusion.
