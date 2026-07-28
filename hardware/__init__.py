"""真机验证脚本：QKCS vs PennyLane default.qubit 一致性测试。

当前为占位模块：原始 `ibmq_verify` 实现已迁出本包，避免被 `import qcc_mamba`
顺带触发不存在的依赖。真机验证请直接调用 `qcc.feature_map.EntanglingFeatureMap`
与 PennyLane 的 `default.qubit` 设备进行对比。
"""
__all__ = []
