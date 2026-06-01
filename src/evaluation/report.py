import json
import datetime
from typing import List, Dict, Optional

from .metrics import compute_fold_stats

class ExperimentReport:

    def __init__(self, experiment_name: str = "anomaly_detection"):
        self.experiment_name = experiment_name
        self.timestamp       = datetime.datetime.now().isoformat()
        self.sections: List[Dict] = []

    def add_section(self, scenario: str, dataset: str,
                    fold_results: List[Dict],
                    stat_test: Optional[Dict] = None) -> None:
        stats = compute_fold_stats(fold_results)
        section = {
            "scenario":   scenario,
            "dataset":    dataset,
            "n_folds":    len(fold_results),
            "stats":      stats,
            "stat_test":  stat_test,
        }
        self.sections.append(section)

    def to_dict(self) -> Dict:
        return {
            "experiment": self.experiment_name,
            "timestamp":  self.timestamp,
            "sections":   self.sections,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent,
                          default=str)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"DENEY RAPORU: {self.experiment_name}")
        print(f"Tarih: {self.timestamp}")
        print(f"{'='*60}")

        header = f"{'Dataset':<10} {'Senaryo':<10} {'F1 Mean':>8} {'F1 Std':>8} {'Acc Mean':>10}"
        print(header)
        print("-" * len(header))

        for s in self.sections:
            f1  = s["stats"].get("f1_score", {})
            acc = s["stats"].get("accuracy", {})
            print(
                f"{s['dataset']:<10} {s['scenario']:<10} "
                f"{f1.get('mean', 'N/A'):>8} {f1.get('std', 'N/A'):>8} "
                f"{acc.get('mean', 'N/A'):>10}"
            )
        print(f"{'='*60}\n")
