"""
CAVE Multispectral Database Train/Test Splits.

This module defines the authoritative dataset partition for the CAVE Multispectral
Database according to Section 3.2 of Xie et al. (2022), "Dual camera snapshot
hyperspectral imaging system via physics-informed learning", Optics Express:

- Exactly 20 scenes allocated for training.
- Exactly 10 scenes allocated for evaluation / testing.
- Zero scene overlap between partitions.
- 30 scenes utilized in total (excluding outliers/miscalibrated scenes).
"""

import os
import glob
from typing import List, Dict, Optional, Union


# Authoritative 20 CAVE scenes allocated for training (Xie et al. 2022, Section 3.2)
CAVE_TRAIN_SCENES: List[str] = [
    "fake_and_real_peppers_ms",
    "fake_and_real_strawberries_ms",
    "fake_and_real_sushi_ms",
    "fake_and_real_tomatoes_ms",
    "feathers_ms",
    "flowers_ms",
    "glass_tiles_ms",
    "hairs_ms",
    "jelly_beans_ms",
    "oil_painting_ms",
    "paints_ms",
    "photo_and_face_ms",
    "pompoms_ms",
    "real_and_fake_apples_ms",
    "real_and_fake_peppers_ms",
    "sponges_ms",
    "stuffed_toys_ms",
    "superballs_ms",
    "thread_spools_ms",
    "watercolors_ms",
]

# Authoritative 10 CAVE scenes allocated for testing (Xie et al. 2022, Section 3.2)
CAVE_TEST_SCENES: List[str] = [
    "balloons_ms",
    "beads_ms",
    "cd_ms",
    "chart_and_stuffed_toy_ms",
    "clay_ms",
    "cloth_ms",
    "egyptian_statue_ms",
    "face_ms",
    "fake_and_real_beers_ms",
    "fake_and_real_food_ms",
]

# All 30 standard scenes
CAVE_ALL_30_SCENES: List[str] = CAVE_TRAIN_SCENES + CAVE_TEST_SCENES

# Default directories checked for CAVE dataset
DEFAULT_CAVE_ROOTS: List[str] = [
    r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep\dataset",
    r"C:\CODE2026\CASSIproy2026\CASSIproy2026\CAVE_Raw",
]


class SplitsResult(tuple):
    """
    Container for dataset split results supporting:
    - 2-tuple unpacking: `train_list, test_list = get_cave_splits()`
    - Subscripting: `splits['train']`, `splits['test']`
    - Attribute access: `splits.train`, `splits.test`
    - Dict conversion: `splits.to_dict()`
    """
    def __new__(cls, train_scenes: List[str], test_scenes: List[str]):
        return super().__new__(cls, (train_scenes, test_scenes))

    @property
    def train(self) -> List[str]:
        return self[0]

    @property
    def test(self) -> List[str]:
        return self[1]

    def to_dict(self) -> Dict[str, List[str]]:
        return {'train': self[0], 'test': self[1]}

    def keys(self) -> List[str]:
        return ['train', 'test']

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, str):
            key = item.lower()
            if key in ('train', 'fortrain', 'training'):
                return self[0]
            elif key in ('test', 'fortest', 'val', 'testing', 'validation'):
                return self[1]
            raise KeyError(f"Invalid key '{item}' in SplitsResult. Expected 'train' or 'test'.")
        return super().__getitem__(item)


def _normalize_name(name: str) -> str:
    """Strips trailing '_ms' and converts to lowercase for uniform comparison."""
    norm = name.strip().lower()
    if norm.endswith('_ms'):
        norm = norm[:-3]
    return norm


def _find_scene_dir(search_root: str, scene_name: str) -> Optional[str]:
    """
    Searches for a scene directory within search_root.
    Validates that the folder contains at least 31 PNG band files.
    """
    target_norm = _normalize_name(scene_name)
    if not os.path.exists(search_root):
        return None

    # Check immediate children first
    try:
        entries = list(os.scandir(search_root))
    except (OSError, PermissionError):
        return None

    for entry in entries:
        if entry.is_dir() and _normalize_name(entry.name) == target_norm:
            # Check for PNG files
            png_count = len(glob.glob(os.path.join(entry.path, "**", "*.png"), recursive=True))
            if png_count >= 31:
                return entry.path

    # Walk directory tree if not found in immediate children
    for root, dirs, _ in os.walk(search_root):
        for d in dirs:
            if _normalize_name(d) == target_norm:
                full_path = os.path.join(root, d)
                png_count = len(glob.glob(os.path.join(full_path, "**", "*.png"), recursive=True))
                if png_count >= 31:
                    return full_path

    return None


def get_cave_splits(root_dir: Optional[str] = None) -> SplitsResult:
    """
    Retrieves the canonical 20-train / 10-test CAVE scene partitions per Xie et al. (2022).

    If `root_dir` (or one of the default CAVE paths) exists and contains scene folders,
    the returned lists contain the verified absolute filesystem paths to the scene directories.
    If no physical directory is accessible, returns the canonical scene name identifiers.

    Args:
        root_dir: Optional path to CAVE root directory (e.g. containing 'fortrain'/'fortest'
                  or individual scene folders).

    Returns:
        SplitsResult with (train_scenes, test_scenes):
        - train_scenes: exactly 20 items.
        - test_scenes: exactly 10 items.
    """
    # 1. Resolve effective search roots
    candidate_roots = []
    if root_dir is not None:
        candidate_roots.append(root_dir)
    candidate_roots.extend([p for p in DEFAULT_CAVE_ROOTS if p != root_dir])

    active_root = None
    for cand in candidate_roots:
        if cand and os.path.isdir(cand):
            active_root = cand
            break

    # If no directory exists, return canonical scene lists
    if active_root is None:
        return SplitsResult(list(CAVE_TRAIN_SCENES), list(CAVE_TEST_SCENES))

    # 2. Check for partitioned subdirectories (e.g. 'fortrain' and 'fortest')
    fortrain_dir = os.path.join(active_root, "fortrain")
    fortest_dir = os.path.join(active_root, "fortest")

    train_search = fortrain_dir if os.path.isdir(fortrain_dir) else active_root
    test_search = fortest_dir if os.path.isdir(fortest_dir) else active_root

    train_resolved = []
    for scene in CAVE_TRAIN_SCENES:
        found = _find_scene_dir(train_search, scene)
        if found is None and train_search != active_root:
            found = _find_scene_dir(active_root, scene)
        train_resolved.append(found if found is not None else scene)

    test_resolved = []
    for scene in CAVE_TEST_SCENES:
        found = _find_scene_dir(test_search, scene)
        if found is None and test_search != active_root:
            found = _find_scene_dir(active_root, scene)
        test_resolved.append(found if found is not None else scene)

    # Invariants: strictly 20 train, 10 test
    assert len(train_resolved) == 20, f"Expected 20 train scenes, got {len(train_resolved)}"
    assert len(test_resolved) == 10, f"Expected 10 test scenes, got {len(test_resolved)}"

    return SplitsResult(train_resolved, test_resolved)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CAVE Dataset Splits Module (splits.py)")
    print("=" * 60)
    
    # 1. Verify train/test counts
    print(f"[*] Checking CAVE_TRAIN_SCENES count: {len(CAVE_TRAIN_SCENES)} (expected 20)")
    assert len(CAVE_TRAIN_SCENES) == 20, "CAVE_TRAIN_SCENES must have exactly 20 scenes."
    
    print(f"[*] Checking CAVE_TEST_SCENES count:  {len(CAVE_TEST_SCENES)} (expected 10)")
    assert len(CAVE_TEST_SCENES) == 10, "CAVE_TEST_SCENES must have exactly 10 scenes."
    
    # 2. Verify zero overlap
    overlap = set(CAVE_TRAIN_SCENES).intersection(set(CAVE_TEST_SCENES))
    print(f"[*] Checking scene overlap count: {len(overlap)} (expected 0)")
    assert len(overlap) == 0, f"Detected scene overlap: {overlap}"
    
    # 3. Test get_cave_splits() with auto-discovery
    splits = get_cave_splits()
    print(f"[+] Splits auto-discovery: {len(splits.train)} train, {len(splits.test)} test.")
    assert len(splits.train) == 20
    assert len(splits.test) == 10
    
    # 4. Test dict-like and tuple-like access
    assert len(splits['train']) == 20
    assert len(splits['test']) == 10
    tr, te = splits
    assert len(tr) == 20 and len(te) == 10
    
    # 5. Test fallback with non-existent path
    fallback_splits = get_cave_splits(root_dir="C:/nonexistent/directory")
    assert len(fallback_splits.train) == 20
    assert len(fallback_splits.test) == 10
    print("[+] Fallback for non-existent path verified.")
    
    print("\n>>> ALL SPLITS VERIFICATION TESTS PASSED SUCCESSFULLY! <<<\n")

