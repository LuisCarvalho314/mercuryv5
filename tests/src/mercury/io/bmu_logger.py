from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
import csv


class BMULogger:
    def __init__(self, save_root: Path, dataset_id: str, run_id: str):
        """
        save_root: base dir, e.g. PROJECT_ROOT / "saved_states"
        dataset_id: dataset / level name
        run_id: rollout id, e.g. "run_0001"
        """
        self.save_root = Path(save_root)
        self.dataset_id = dataset_id
        self.run_id = run_id

        # store (t, bmu, bmu_gt_optional)
        self._rows: List[Tuple[int, int, Optional[int]]] = []

    def push(self, t: int, bmu: int, gt_bmu: Optional[int] = None) -> None:
        """
        t: timestep index (int)
        bmu: current model BMU (int)
        gt_bmu: optional ground truth id (int or None)
        """
        self._rows.append((int(t), int(bmu), None if gt_bmu is None else int(gt_bmu)))

    def save(self) -> None:
        """
        Write CSV:
            t,bmu,bmu_gt
            1,12,
            2,12,5
            3,7,5
        """
        out_dir = self.save_root / self.dataset_id / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "bmu_sequence.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "bmu", "bmu_gt"])
            for t, bmu, gt in self._rows:
                writer.writerow([t, bmu, "" if gt is None else gt])
