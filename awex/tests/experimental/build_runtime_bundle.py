#!/usr/bin/env python3

import argparse
import tarfile
from collections import defaultdict, deque
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def dependency_closure(root_distribution):
    requested_extras = defaultdict(set)
    requested_extras[canonicalize_name(root_distribution)] = set()
    queue = deque([root_distribution])
    distributions = {}

    while queue:
        name = queue.popleft()
        canonical_name = canonicalize_name(name)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            print(f"skipped unavailable dependency {name}")
            continue
        distributions[canonical_name] = distribution
        active_extras = requested_extras[canonical_name]

        for requirement_text in distribution.requires or ():
            requirement = Requirement(requirement_text)
            marker_contexts = ({"extra": ""},) + tuple(
                {"extra": extra} for extra in active_extras
            )
            if requirement.marker and not any(
                requirement.marker.evaluate(context) for context in marker_contexts
            ):
                continue

            dependency_name = canonicalize_name(requirement.name)
            dependency_extras = set(requirement.extras)
            is_new = dependency_name not in distributions
            extras_changed = not dependency_extras.issubset(
                requested_extras[dependency_name]
            )
            requested_extras[dependency_name].update(dependency_extras)
            if is_new or extras_changed:
                queue.append(requirement.name)

    return distributions


def is_excluded(name, prefixes):
    return any(name == prefix or name.startswith(f"{prefix}-") for prefix in prefixes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-distribution", default="vllm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Canonical distribution-name prefix to leave to the target environment.",
    )
    args = parser.parse_args()

    excluded = {canonicalize_name(prefix) for prefix in args.exclude_prefix}
    distributions = dependency_closure(args.root_distribution)
    root_name = canonicalize_name(args.root_distribution)
    selected = {
        name: distribution
        for name, distribution in distributions.items()
        if name == root_name or not is_excluded(name, excluded)
    }

    archived_paths = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w") as archive:
        for name, distribution in sorted(selected.items()):
            base = Path(distribution.locate_file("")).resolve()
            for relative_path in distribution.files or ():
                source = Path(distribution.locate_file(relative_path)).resolve()
                try:
                    archive_path = source.relative_to(base)
                except ValueError:
                    continue
                if archive_path in archived_paths or not source.exists():
                    continue
                archive.add(source, arcname=archive_path, recursive=False)
                archived_paths.add(archive_path)
            print(f"included {name}=={distribution.version}")

    print(
        f"wrote {args.output} with {len(selected)} distributions and "
        f"{len(archived_paths)} files"
    )


if __name__ == "__main__":
    main()
