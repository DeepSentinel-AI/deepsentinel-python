# Versioned Documentation Release Guide

This guide explains how to manage and release versioned documentation for the DeepSentinel Python SDK.

## Overview

The documentation system uses [Mike](https://github.com/jimporter/mike) to manage versioned documentation deployments to GitHub Pages. The workflow supports:

- **Latest docs**: Main branch docs are automatically deployed to `/latest`
- **Versioned docs**: Tagged releases create versioned documentation at `/v{version}`
- **Default redirect**: The `/latest` path serves as the default landing page

## Automatic Deployment

### Main Branch (Latest)
- **Trigger**: Push to `main` branch with changes to `docs/` or `mkdocs.yml`
- **Deployment**: Automatically deployed to `/latest`
- **URL**: `https://deepsentinel-ai.github.io/deepsentinel-python/latest/`

### Version Tags
- **Trigger**: Push a version tag (e.g., `v1.0.0`, `v1.2.3`)
- **Deployment**: Automatically deployed to `/v{version}` 
- **URL**: `https://deepsentinel-ai.github.io/deepsentinel-python/v1.0.0/`

## Manual Release Process

### 1. Prepare for Release

Ensure your documentation is ready:
```bash
cd python/
# Build and test locally
mkdocs serve
# Visit http://localhost:8000 to verify
```

### 2. Create Version Tag

```bash
# Tag the release (replace X.Y.Z with actual version)
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

### 3. Automatic Deployment

The GitHub Actions workflow will automatically:
1. Detect the version tag
2. Extract version number (e.g., `v1.0.0` → `1.0.0`)
3. Deploy docs to versioned path
4. Update the version list

### 4. Manual Deployment (if needed)

If automatic deployment fails, you can deploy manually:

```bash
# Install mike
pip install mike

# Deploy specific version
mike deploy --push --update-aliases 1.0.0

# Set as default (optional)
mike set-default --push 1.0.0

# List all versions
mike list
```

## Version Management Commands

### List Available Versions
```bash
mike list
```

### Delete a Version
```bash
mike delete --push old-version
```

### Set Default Version
```bash
mike set-default --push 1.0.0
```

### Deploy with Alias
```bash
# Deploy version with alias
mike deploy --push --update-aliases 1.0.0 stable
```

## URL Structure

| Branch/Tag | URL Path | Example |
|------------|----------|---------|
| `main` | `/latest` | `https://deepsentinel-ai.github.io/deepsentinel-python/latest/` |
| `v1.0.0` | `/v1.0.0` | `https://deepsentinel-ai.github.io/deepsentinel-python/v1.0.0/` |
| `v1.1.0` | `/v1.1.0` | `https://deepsentinel-ai.github.io/deepsentinel-python/v1.1.0/` |

## Best Practices

### 1. Version Naming
- Use semantic versioning: `vMAJOR.MINOR.PATCH`
- Examples: `v1.0.0`, `v1.2.3`, `v2.0.0-beta.1`

### 2. Documentation Updates
- Update version-specific content before tagging
- Include version in changelog and release notes
- Test documentation locally before releasing

### 3. Release Workflow
1. Complete development and testing
2. Update documentation for the release
3. Create and push version tag
4. Verify automatic deployment
5. Announce the release

## Troubleshooting

### Deployment Fails
1. Check GitHub Actions logs
2. Verify `mike` installation in workflow
3. Ensure proper permissions are set
4. Try manual deployment as fallback

### Version Not Showing
1. Verify tag format matches `v*.*.*`
2. Check if workflow was triggered
3. Confirm `mike list` shows the version

### Manual Cleanup
If you need to clean up the gh-pages branch:
```bash
# Checkout gh-pages branch
git checkout gh-pages

# List versions
mike list

# Remove unwanted version
mike delete unwanted-version --push
```

## Integration with CI/CD

The workflow integrates with:
- **GitHub Actions**: Automatic deployment on tag push
- **GitHub Pages**: Hosting platform
- **MkDocs**: Documentation generator
- **Mike**: Version management

## Support

For issues with documentation deployment:
1. Check the [GitHub Actions workflow logs](https://github.com/DeepSentinel-AI/deepsentinel-python/actions)
2. Verify your local MkDocs setup works: `mkdocs serve`
3. Test mike commands locally before pushing tags
4. Contact the development team for persistent issues