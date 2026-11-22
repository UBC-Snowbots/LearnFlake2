#!/usr/bin/env bash
# Repo hygiene check, used by CI and runnable locally.
#
#   scripts/check_hygiene.sh [REV | BASE..HEAD]
#
# With REV (default HEAD) every commit reachable from REV is checked. With BASE..HEAD only that range is.
# Fails when:
#   - a commit message has a "Co-authored-by:" line
#   - a commit message has a "Generated with" line
#   - a file in the tree at the end of the range is larger than 5 MB (weights and datasets go to Releases)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
spec="${1:-HEAD}"
max_bytes=$((5 * 1024 * 1024))
fail=0

case "$spec" in
  *..*) range="$spec"; tip="${spec##*..}"; [ -n "$tip" ] || tip=HEAD ;;
  *) range="$spec"; tip="$spec" ;;
esac

bad_msg='^[[:space:]]*(co-authored-by:|.*generated with)'
for c in $(git rev-list "$range"); do
  if git show -s --format=%B "$c" | grep -i -q -E "$bad_msg"; then
    echo "FAIL: commit $(git rev-parse --short "$c") has a Co-authored-by or Generated-with line:"
    git show -s --format=%B "$c" | grep -i -n -E "$bad_msg" | sed 's/^/    /'
    fail=1
  fi
done

while read -r size path; do
  if [ "$size" -gt "$max_bytes" ]; then
    echo "FAIL: $path is $size bytes (limit $max_bytes); publish it as a release asset instead"
    fail=1
  fi
done < <(git ls-tree -r -l "$tip" | awk '$2 == "blob" { size = $4; sub(/^[^\t]*\t/, ""); print size, $0 }')

if [ "$fail" -eq 0 ]; then
  echo "hygiene OK ($range)"
fi
exit "$fail"
