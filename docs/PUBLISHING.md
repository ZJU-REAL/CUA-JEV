# Publishing the CUA-JEV website

This is a maintainer note, not part of the public quick start. [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) builds a read-only GitHub Pages site from `main` and deploys it to <https://zjureal.com/CUA-JEV/>.

The build reads only reviewed [`website/snapshot.json`](../website/snapshot.json) and [`website/media/`](../website/media/). It does not upload local `runs/`, `.env`, or raw recordings. After new experiments, inspect the data and video copies locally before running:

```powershell
python scripts/build_pages.py --refresh-snapshot
python scripts/prepare_public_demos.py
python scripts/build_pages.py --build
```

Review the `website/` diff for personal information, credentials, local paths, and unintended footage before committing and pushing. Updates to `main` deploy automatically through GitHub Actions. See the [GitHub Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
