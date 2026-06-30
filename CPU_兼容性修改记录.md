# CPU 兼容性修改记录

> 目的：让 `infer_image_from_mask.py` (MAISI Step 3) 在无 GPU 的 CPU 机器上运行。
> 修改日期：2026-06-29
> 修改范围：1 个文件，3 处

---

## 修改文件：`scripts/utils_infer.py`

### 修改 1：autocast 设备类型（第 197 行）

**原代码：**
```python
with torch.no_grad(), torch.amp.autocast("cuda"):
```

**改为：**
```python
with torch.no_grad(), torch.amp.autocast(device.type):
```

**原因：** 硬编码 `"cuda"` 在 CPU 上会抛出 RuntimeError。改为 `device.type` 后，GPU 上使用 CUDA autocast，CPU 上自动降级为 no-op。

---

### 修改 2：去噪循环后的显存清理（第 291 行）

**原代码：**
```python
        gc.collect()
        torch.cuda.empty_cache()

        # Sliding-window AE decode
```

**改为：**
```python
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Sliding-window AE decode
```

**原因：** CPU 上没有 CUDA context，调用 `torch.cuda.empty_cache()` 会报错。

---

### 修改 3：图像解码后的显存清理（第 319 行）

**原代码：**
```python
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        torch.cuda.empty_cache()
```

**改为：**
```python
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        if device.type == "cuda":
            torch.cuda.empty_cache()
```

**原因：** 同修改 2。

---

## 快速恢复

```powershell
cd c:\Users\33067\.claude\work\MAISI
git apply cpu_fix.patch
```

---

## 未修改的文件（无需处理）

| 文件 | 原因 |
|------|------|
| `scripts/inference.py` | Step 1（mask 生成），GPU 端完成 |
| `scripts/sample_mask.py` | DDPM mask 生成，Step 1 内部 |
| `scripts/tumor_prompt_runner.py` | DiffTumor 管线，不使用 |
| `scripts/*train*.py` | 训练脚本，不使用 |
