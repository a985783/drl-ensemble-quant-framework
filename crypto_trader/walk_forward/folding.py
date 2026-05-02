"""
FoldSpec 配置生成器
为 Walk-Forward 验证生成标准的折叠时间分割配置

每个 FoldSpec 包含：
- fold_id: 折叠标识
- train_start/train_end: 训练期起止
- test_start/test_end: 测试期起止（严格不重叠）
- val_split_date: 训练期内 80% 时间点，用于门控 temperature 选择
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List


@dataclass
class FoldConfig:
    """单个折叠的时间配置"""
    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    val_split_date: str  # 训练期 80% 分位点，用于门控 temperature 选择


class FoldingManager:
    """从配置构建并验证 FoldSpec 列表"""

    def __init__(self, config: Any):
        """
        Args:
            config: 包含 folds 属性的对象，或 fold 列表本身
                    每个 fold 为 dict，包含:
                    fold_id, train_start, train_end, test_start, test_end
        """
        self.config = config

    def build_folds(self) -> List[FoldConfig]:
        """构建 5 个 FoldSpec，自动计算 val_split_date"""
        # 兼容传入 fold 列表或 config.folds 两种方式
        raw_folds: List[Dict] = (
            self.config if isinstance(self.config, list)
            else self.config.folds
        )

        folds: List[FoldConfig] = []
        for f in raw_folds:
            train_start = f.train_start if hasattr(f, 'train_start') else f["train_start"]
            train_end = f.train_end if hasattr(f, 'train_end') else f["train_end"]
            test_start = f.test_start if hasattr(f, 'test_start') else f["test_start"]
            test_end = f.test_end if hasattr(f, 'test_end') else f["test_end"]
            fold_id = f.fold_id if hasattr(f, 'fold_id') else f.get("fold_id", f.get("name", "unknown"))

            train_start_dt = datetime.strptime(train_start, "%Y-%m-%d")
            train_end_dt = datetime.strptime(train_end, "%Y-%m-%d")
            train_days = (train_end_dt - train_start_dt).days

            val_offset = max(1, int(train_days * 0.8))
            val_split_dt = train_start_dt + timedelta(days=val_offset)

            fold = FoldConfig(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                val_split_date=val_split_dt.strftime("%Y-%m-%d"),
            )
            folds.append(fold)

        self.validate_folds(folds)
        return folds

    def validate_folds(self, folds: List[FoldConfig]) -> bool:
        """验证所有折叠：测试期必须严格在训练期之后，不可重叠"""
        for fold in folds:
            train_end_dt = datetime.strptime(fold.train_end, "%Y-%m-%d")
            test_start_dt = datetime.strptime(fold.test_start, "%Y-%m-%d")
            if test_start_dt <= train_end_dt:
                raise ValueError(
                    f"{fold.fold_id}: test_start {fold.test_start} <= "
                    f"train_end {fold.train_end} (overlap not allowed)"
                )
        return True

    def describe(self, folds: List[FoldConfig]) -> str:
        """返回折叠配置的人类可读摘要"""
        lines = ["Fold Summary:", "─" * 80]
        for fold in folds:
            lines.append(
                f"  {fold.fold_id:<20} train: {fold.train_start} → {fold.train_end}  "
                f"| test: {fold.test_start} → {fold.test_end}  "
                f"| val_split: {fold.val_split_date}"
            )
        lines.append("─" * 80)
        return "\n".join(lines)

    @staticmethod
    def from_existing_folds(folds: List[Dict]) -> "FoldingManager":
        """从现有 FOLDS 格式（无 train_start）构建 manager，
        使用固定的系统起始日期"""
        SYSTEM_START = "2020-01-01"
        augmented = []
        for f in folds:
            augmented.append({
                "fold_id": f["name"],
                "train_start": SYSTEM_START,
                "train_end": f["train_end"],
                "test_start": f["test_start"],
                "test_end": f["test_end"],
            })
        return FoldingManager(augmented)
