"""训练 / 评估 / 统计。"""
from .train import train_one_epoch, evaluate, fit, build_optimizer, set_global_stats
from .evaluate import paired_t_test, metric_table, format_results

__all__ = [
    "train_one_epoch",
    "evaluate",
    "fit",
    "build_optimizer",
    "set_global_stats",
    "paired_t_test",
    "metric_table",
    "format_results",
]
