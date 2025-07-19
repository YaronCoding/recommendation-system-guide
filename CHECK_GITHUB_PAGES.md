# GitHub Pages 设置检查清单

您需要在GitHub仓库中完成以下设置来启用Pages：

## 🔧 必需的设置步骤：

### 1. 启用 GitHub Pages
1. 打开您的GitHub仓库：https://github.com/YaronCoding/recommendation-system-guide
2. 点击 **Settings**（设置）标签
3. 在左侧菜单中找到 **Pages**
4. 在 **Source** 部分选择 **GitHub Actions**
5. 点击 **Save**（保存）

### 2. 检查 Actions 权限
1. 在 Settings 中，点击左侧的 **Actions** > **General**
2. 确保 **Workflow permissions** 设置为：
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**

### 3. 检查 Actions 运行状态
1. 点击仓库的 **Actions** 标签
2. 查看最新的 "部署文档" 工作流是否成功运行
3. 如果失败，点击查看错误日志

### 4. 验证部署
部署成功后，您的文档将在以下地址可用：
```
https://yaroncoding.github.io/recommendation-system-guide/
```

## 🔍 常见问题排查：

### 问题1：Actions 没有运行
- 检查 `.github/workflows/docs.yml` 文件是否存在
- 确保推送到了 `main` 分支

### 问题2：Actions 运行失败
- 查看 Actions 日志中的具体错误信息
- 常见错误：Python依赖安装失败、MkDocs构建失败

### 问题3：Pages 显示404
- 确认 Pages 源设置为 "GitHub Actions"
- 等待几分钟让部署完成
- 检查是否有 DNS 缓存问题

## 📞 需要帮助？
如果遇到问题，请提供：
1. GitHub Actions 的运行日志
2. GitHub Pages 设置的截图
3. 具体的错误信息 