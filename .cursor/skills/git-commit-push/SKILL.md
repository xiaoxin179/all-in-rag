---
name: git-commit-push
description: 简化 Git 提交流程，自动执行 git add、commit 和 push。适用于用户要求提交代码、推送代码到远程仓库时使用。
---

# Git 提交并推送

## 工作流程

当用户要求提交代码时，执行以下步骤：

### 1. 自动执行 Git 命令（无需询问用户）

由于用户已确认要提交代码，直接依次执行以下三条命令：

**重要**：PowerShell 不支持 `&&` 语法，必须分开执行。使用分号 `;` 连接命令，或使用三个独立的命令调用。

```powershell
git add .
git commit -m "用户的 commit message"
git push origin main
```

### 2. 验证结果

- 如果 push 成功，显示成功信息
- 如果 push 失败（如未连接到远程、权限问题），提示用户具体错误

## 执行方式

| Shell | 命令格式 |
|-------|---------|
| PowerShell (Windows) | `git add .; git commit -m "..."; git push origin main` |
| Bash / Zsh | `git add . && git commit -m "..." && git push origin main` |

## 注意事项

- **仅限此三条命令自动执行**：git add、git commit、git push
- 其他 git 命令（如 `git status`、`git log`、`git diff` 等）仍需询问用户是否同意执行
- commit message 使用用户提供的原始文本
- 使用 `origin main` 作为默认远程分支
