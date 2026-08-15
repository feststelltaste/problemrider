#!/usr/bin/env python3
"""
Generates the data file behind the /landscape/ page: a 2D layout of every
problem and every solution, positioned by semantic similarity instead of by
an editorially-drawn causal graph.

How it works:
    1. Load each problem's/solution's cached embedding vector from
       embeddings/problems/<slug>.yaml resp. embeddings/solutions/<slug>.yaml
       (produced by calculate_related_problems.py / calculate_related_solutions.py).
    2. Reduce the embeddings to 2D with UMAP (cosine metric), so items with a
       similar meaning end up near each other. UMAP was chosen over classical
       MDS/t-SNE because it keeps some global structure (unlike t-SNE, whose
       cluster distances are meaningless) while still separating clusters
       much more crisply than MDS, and is deterministic given a fixed
       random_state.
    3. Run k-means on the original embeddings to find a fixed number of
       semantic groups, then push each group further away from the overall
       center in whatever direction it already sits in the 2D layout —
       otherwise UMAP alone tends to produce one continuous haze rather than
       legible, separated clusters (confirmed by trying density-based
       clustering first: it collapsed 80%+ of one collection into a single
       mega-cluster instead of finding distinct groups).
    4. Nudge apart nodes that ended up so close their (multi-line) text
       labels would overlap, while gently springing back toward the
       cluster-separated position so the overall clustering survives the
       nudging.
    5. Items that have no cached embedding yet (e.g. a brand-new entry
       before the next embedding run) fall back to the average position of
       whichever of their `related_problems`/`related_solutions` neighbors
       do have one.

Usage:
    python scripts/create_landscape.py
    python scripts/create_landscape.py --separation 3.2 --min-label-distance 130

Output:
    assets/js/landscape-data.js — plain JS assigning window.LANDSCAPE_DATA,
    loaded by landscape.html / assets/js/landscape.js.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans
from umap import UMAP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Virtual canvas the (x, y) coordinates are laid out on. The frontend uses
# these as plain pixel coordinates inside a pannable/zoomable surface, so
# picking a generous size here is what gives the multi-line text labels
# room to breathe.
CANVAS_WIDTH = 2400
CANVAS_HEIGHT = 1600
# How much further apart the k-means clusters are pushed from the overall
# center, on top of whatever gap UMAP already produced between them (1.0
# leaves UMAP's raw layout untouched). Overridable with --separation.
DEFAULT_SEPARATION_FACTOR = 3.0
# Minimum center-to-center distance (in canvas px) two labels are pushed
# apart to, so a two/three-line title has a decent chance of staying
# legible. Overridable with --min-label-distance.
DEFAULT_MIN_LABEL_DISTANCE = 130


def parse_frontmatter(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        return yaml.safe_load(parts[1])
    except yaml.YAMLError as error:
        print(f"⚠️  Could not parse frontmatter of {path}: {error}")
        return None


def primary_category(frontmatter):
    category = frontmatter.get("category")
    if isinstance(category, list) and category:
        return category[0]
    if isinstance(category, str) and category:
        return category
    return "Uncategorized"


def load_items(items_dir, embeddings_dir, related_field):
    items = []
    pattern = os.path.join(PROJECT_ROOT, items_dir, "*.md")
    for path in sorted(glob.glob(pattern)):
        slug = Path(path).stem
        frontmatter = parse_frontmatter(path)
        if not frontmatter or "title" not in frontmatter:
            continue

        embedding_path = os.path.join(PROJECT_ROOT, embeddings_dir, f"{slug}.yaml")
        embedding = None
        if os.path.exists(embedding_path):
            with open(embedding_path, "r", encoding="utf-8") as f:
                cache = yaml.safe_load(f)
            raw = cache.get("embedding") if cache else None
            if raw:
                embedding = np.array(raw, dtype=float)

        items.append(
            {
                "slug": slug,
                "title": frontmatter["title"],
                "category": primary_category(frontmatter),
                "embedding": embedding,
                "related": frontmatter.get(related_field) or [],
            }
        )
    return items


def cluster_and_separate(coords, matrix, label, separation_factor):
    """UMAP alone tends to produce one continuous haze rather than legible
    groups — density-based clustering (HDBSCAN) on that haze confirmed it,
    collapsing 80%+ of one collection into a single mega-cluster instead of
    finding distinct groups. K-means on the original embeddings sidesteps
    that: a fixed cluster count guarantees the map actually reads as several
    groups. Each cluster is then pushed further away from the overall center
    in whatever direction it already sits in the 2D layout — amplifying gaps
    that already exist between groups instead of inventing an arrangement
    unrelated to what UMAP produced."""
    n = len(coords)
    # Roughly one cluster per 25 items, so the map reads as legible groups
    # without fragmenting into clusters of one or two.
    n_clusters = max(6, round(n / 25))
    cluster_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(matrix)

    sizes = np.bincount(cluster_labels)
    print(f"{label}: {n_clusters} clusters (sizes {sizes.min()}-{sizes.max()}, mean {sizes.mean():.0f}), separation={separation_factor}")

    center = coords.mean(axis=0)
    separated = coords.copy()
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        centroid = coords[mask].mean(axis=0)
        separated[mask] += (centroid - center) * (separation_factor - 1)
    return separated, cluster_labels


def declutter(coords, min_label_distance):
    """Iteratively push apart points closer than min_label_distance, with a
    weak spring back toward the original UMAP position so clusters spread out
    locally without losing their overall place in the layout."""
    coords = coords.copy()
    original = coords.copy()
    spring = 0.02
    for _ in range(300):
        diff = coords[:, None, :] - coords[None, :, :]  # n x n x 2
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        overlap = np.clip(min_label_distance - dist, 0, None)
        if not overlap.any():
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            direction = diff / dist[..., None]
        direction = np.nan_to_num(direction)
        push = (direction * overlap[..., None]).sum(axis=1)
        coords += push * 0.5
        coords += (original - coords) * spring
    return coords


def layout(items, label, separation_factor, min_label_distance):
    with_embedding = [item for item in items if item["embedding"] is not None]
    without_embedding = [item for item in items if item["embedding"] is None]
    print(f"{label}: {len(with_embedding)} with cached embedding, {len(without_embedding)} without")

    positions = {}
    if with_embedding:
        matrix = np.vstack([item["embedding"] for item in with_embedding])
        # n_neighbors/min_dist are tuned lower than UMAP's defaults (15/0.1)
        # to favor tight, visually distinct clusters over preserving finer
        # continuous structure — this is a map meant to be read at a glance,
        # not a scientific embedding.
        n_neighbors = max(2, min(15, len(with_embedding) - 1))
        reducer = UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=0.05,
            random_state=42,
        )
        coords = reducer.fit_transform(matrix)
        coords, cluster_labels = cluster_and_separate(coords, matrix, label, separation_factor)
        for item, cluster_id in zip(with_embedding, cluster_labels):
            item["cluster"] = int(cluster_id)

        # Scale to the virtual canvas before decluttering, so the minimum
        # label distance is expressed in the same units as the canvas.
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-9)
        margin = 80
        scaled = (coords - min_xy) / span
        scaled[:, 0] = scaled[:, 0] * (CANVAS_WIDTH - 2 * margin) + margin
        scaled[:, 1] = scaled[:, 1] * (CANVAS_HEIGHT - 2 * margin) + margin
        scaled = declutter(scaled, min_label_distance)

        for item, (x, y) in zip(with_embedding, scaled):
            positions[item["slug"]] = (float(x), float(y))

    # Items without their own embedding yet: average the position of
    # whichever related neighbors already have one; otherwise drop them in
    # the center of the canvas rather than silently discarding the node.
    for item in without_embedding:
        neighbours = [
            positions[related["slug"]]
            for related in item["related"]
            if isinstance(related, dict) and related.get("slug") in positions
        ]
        if neighbours:
            xs, ys = zip(*neighbours)
            positions[item["slug"]] = (sum(xs) / len(xs), sum(ys) / len(ys))
        else:
            positions[item["slug"]] = (CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)

    nodes = []
    for item in items:
        x, y = positions[item["slug"]]
        nodes.append(
            {
                "id": item["slug"],
                "title": item["title"],
                "category": item["category"],
                "cluster": item.get("cluster"),
                "x": round(x, 1),
                "y": round(y, 1),
            }
        )
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--separation",
        type=float,
        default=DEFAULT_SEPARATION_FACTOR,
        help=f"How much further apart clusters are pushed from the center (default {DEFAULT_SEPARATION_FACTOR}). "
        "Higher = more distance between cluster groups.",
    )
    parser.add_argument(
        "--min-label-distance",
        type=float,
        default=DEFAULT_MIN_LABEL_DISTANCE,
        help=f"Minimum center-to-center spacing between labels in canvas px (default {DEFAULT_MIN_LABEL_DISTANCE}). "
        "Higher = more room between individual texts.",
    )
    args = parser.parse_args()

    problems = load_items("_problems", "embeddings/problems", "related_problems")
    solutions = load_items("_solutions", "embeddings/solutions", "related_solutions")

    data = {
        "canvas": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
        "problems": layout(problems, "Problems", args.separation, args.min_label_distance),
        "solutions": layout(solutions, "Solutions", args.separation, args.min_label_distance),
    }

    js_content = "window.LANDSCAPE_DATA = " + json.dumps(data, indent=2) + ";\n"
    output_path = os.path.join(PROJECT_ROOT, "assets", "js", "landscape-data.js")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"Wrote {len(data['problems'])} problems and {len(data['solutions'])} solutions to {output_path}")


if __name__ == "__main__":
    main()
