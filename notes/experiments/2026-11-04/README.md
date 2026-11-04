# 11 月 4 日实验记录

- [完整验证](full-validation.json)覆盖 5,465,882 个目标位置。
- [温度 0.5](temperature/0.5.json)、[温度 0.8](temperature/0.8.json)和[温度 1.0](temperature/1.0.json)保留生成文本、参数及 token ID。
- [学习率汇总](lr-stability/summary.json)、[峰值学习率 3](lr-stability/lr_3_loss.csv)和[峰值学习率 10](lr-stability/lr_10_loss.csv)保留失稳前后的验证记录。
- [网页分词器小样本对照](owt-tokenizer-pilot.json)包含 6 组逐篇字节数、token 数和还原结果。
- [50 MiB](owt-scale/benchmark-50m.json)与[500 MiB](owt-scale/benchmark-500m.json)是堆版基线，[50 MiB 紧凑版](owt-scale/benchmark-50m-compact.json)与[500 MiB 紧凑版](owt-scale/benchmark-500m-compact.json)是节省内存后的结果。

[当天笔记](../../2026-11-04-A1-EVALUATION-AND-OWT-BPE.md)
