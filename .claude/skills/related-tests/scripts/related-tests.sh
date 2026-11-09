#!/usr/bin/env bash
# Picks the test targets a change can affect and (with --run) runs only those.
#
#   related-tests.sh                    working tree vs HEAD (+ untracked files)
#   related-tests.sh --base origin/main working tree vs a ref
#   related-tests.sh --files a.swift b  an explicit list of repo-relative paths
#   add --run to execute xcodebuild test (Debug) with -only-testing per target
#
# The target graph comes from `swift package dump-package`, so it follows
# Package.swift with no hand-kept table.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

base=HEAD
run=0
explicit=0
files=()
while (($#)); do
  case "$1" in
    --base) base="$2"; shift 2 ;;
    --run) run=1; shift ;;
    --files) explicit=1; shift
      while (($#)) && [[ "$1" != --* ]]; do files+=("$1"); shift; done ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ((!explicit)); then
  while IFS= read -r f; do [[ -n "$f" ]] && files+=("$f"); done < <(
    { git diff --name-only "$base"; git ls-files --others --exclude-standard; } | sort -u)
fi

if ((${#files[@]} == 0)); then
  echo "No changed files (compared with $base)."
  exit 0
fi

# --disable-sandbox: SwiftPM's own sandbox-exec can't nest inside another sandbox.
if ! pkg=$(swift package --disable-sandbox dump-package 2>/dev/null); then
  echo "error: swift package dump-package failed; run it by hand to see why" >&2
  exit 2
fi

report=$(jq -c --args '
  . as $p
  | [$p.targets[] | {
      name, type,
      path: (.path // ((if .type == "test" then "Tests/" else "Sources/" end) + .name)),
      deps: [.dependencies[] | (.byName // .target // empty) | .[0]]
    }] as $t
  | ($t | map(.name)) as $local
  | ($t | map({key: .name, value: [.deps[] | select(IN($local[]))]}) | from_entries) as $g
  | def closure($n): reduce ($g[$n][]) as $d ([]; . + [$d] + closure($d)) | unique;
  ($t | map(select(.type == "test") | .name) | sort) as $tests
  | ($tests | map({key: ., value: ([.] + closure(.))}) | from_entries) as $reach
  | [$ARGS.positional[] as $f
      | if ($f == "Package.swift" or $f == "Package.resolved") then {file: $f, target: "*"}
        else ([$t[] | select(.path as $pp | $f | startswith($pp + "/"))] | max_by(.path | length)) as $m
          | {file: $f, target: ($m.name // null)}
        end] as $map
  | ($map | map(select(.target != null and .target != "*")) | map(.target) | unique) as $changed
  | (any($map[]; .target == "*")) as $all
  | {
      all: $all,
      byTarget: ($map | map(select(.target != null and .target != "*")) | group_by(.target)
                 | map({key: .[0].target, value: map(.file)}) | from_entries),
      uncovered: ($map | map(select(.target == null) | .file)),
      tests: (if $all then $tests
              else [$tests[] as $x | select(any($reach[$x][]; IN($changed[]))) | $x] end),
      untested: [$changed[] as $c | select([$reach[][] | select(. == $c)] | length == 0) | $c]
    }
' "${files[@]}" <<<"$pkg")

echo "Changed files by target:"
jq -r '.byTarget | to_entries[] | "  \(.key):", (.value[] | "    \(.)")' <<<"$report"
if [[ $(jq '.uncovered | length' <<<"$report") -gt 0 ]]; then
  echo "Not in any target (no tests cover these):"
  jq -r '.uncovered[] | "    \(.)"' <<<"$report"
fi
if [[ $(jq '.untested | length' <<<"$report") -gt 0 ]]; then
  echo "Changed targets with no unit tests (consider a smoke test):"
  jq -r '.untested[] | "    \(.)"' <<<"$report"
fi
[[ $(jq '.all' <<<"$report") == true ]] && echo "Package.swift/Package.resolved changed: selecting every test target."

tests=()
while IFS= read -r t; do [[ -n "$t" ]] && tests+=("$t"); done < <(jq -r '.tests[]' <<<"$report")
if ((${#tests[@]} == 0)); then
  echo "Related test targets: none."
  exit 0
fi
echo "Related test targets: ${tests[*]}"

cmd=(xcodebuild test -scheme MLXComfyUI-Package -destination 'platform=macOS'
     -derivedDataPath .build/xcode -skipPackagePluginValidation -skipMacroValidation)
for t in "${tests[@]}"; do cmd+=("-only-testing:$t"); done
printf 'Command:\n  %s\n' "${cmd[*]}"
((run)) || exit 0

log="${TMPDIR:-/tmp}/related-tests.log"
echo "Running (log: $log)…"
start=$SECONDS
set +e
"${cmd[@]}" >"$log" 2>&1
status=$?
set -e
grep -E '(✘|error:|Test run with|Suite .* (passed|failed)|\*\* TEST (SUCCEEDED|FAILED) \*\*)' "$log" \
  | grep -vE '^\s*$' | tail -40 || true
echo "xcodebuild exit $status after $((SECONDS - start)) s"
exit $status
