# Uploading this release to the existing GitHub repository

Repository: `https://github.com/sokikahio-max/FactoryAI-Resilience-Dashboard`

## Recommended safe workflow on Windows PowerShell

### 1. Clone the existing repository into a new folder

```powershell
cd C:\
git clone https://github.com/sokikahio-max/FactoryAI-Resilience-Dashboard.git FactoryAI-Resilience-Dashboard
cd C:\FactoryAI-Resilience-Dashboard
```

### 2. Check the active branch

```powershell
git branch --show-current
```

Usually this will be `main`.

### 3. Keep the hidden `.git` folder, but replace the old project files

Copy the contents of this GitHub-ready package into `C:\FactoryAI-Resilience-Dashboard`, replacing the old project files. Do **not** delete the hidden `.git` directory.

### 4. Inspect changes before committing

```powershell
git status
git diff --stat
```

### 5. Commit the manuscript-aligned v7 release

```powershell
git add -A
git commit -m "Update repository to manuscript-aligned failure-aware LLM-AutoML v7"
git push origin main
```

If your active branch is not `main`, replace `main` with the value returned by `git branch --show-current`.

### 6. Create a GitHub release

On GitHub: Releases → Draft a new release → create tag `v1.0.0-manuscript` → title `Manuscript reproducibility release` → publish the release.

After that release is public, archive the release in Zenodo (or update the existing Zenodo record) and use the DOI that corresponds to this exact release in the manuscript.
