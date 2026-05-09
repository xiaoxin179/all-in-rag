---
name: git-commit-push
description: 简化 Git 提交流程，自动执行 git add、commit 和 push。适用于用户要求提交代码、推送代码到远程仓库时使用。
---

# Git 提交并推送

## 工作流程

当用户要求提交代码时，执行以下步骤：

### 1. 确认提交信息

用户需要提供 commit message。如果没有提供，询问用户。

### 2. 执行 Git 命令

在项目根目录（`d:\selfdevapp\ai\all-in-rag`）依次执行：

```bash
git add .
git commit -m "用户的 commit message"
git push origin main
```

### 3. 验证结果

- 如果 push 成功，显示成功信息
- 如果 push 失败（如未连接到远程、权限问题），提示用户具体错误

## 注意事项

- 始终使用 `git add .` 添加所有更改
- commit message 使用用户提供的原始文本
- 使用 `origin main` 作为默认远程分支

## 示例对话

**用户**: "帮我提交代码"
**Agent**: "请提供本次提交的描述信息："
**用户**: "添加 RAG 入门示例代码"
**Agent**: 执行 git add . → git commit → git push，显示结果
