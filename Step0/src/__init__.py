# Tumor Mask Generator
# 肿瘤位置Mask自动生成器
#
# 项目结构:
#   utils.py              - 工具函数 (坐标变换/形态学操作/HU处理)
#   data_loader.py         - 数据加载 (CT + 器官mask)
#   validator.py           - 校验模块 (位置/mask质量)
#   position_selector.py   - 位置选择 (多种策略)
#   mask_generator.py      - Mask生成 (椭球 + 弹性形变 + 后处理)
#   main.py                - 主入口 (批量生成)
