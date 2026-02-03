name: Generate site (DeepSeek)

on:
  workflow_dispatch:
  schedule:
    - cron: "0 */8 * * *"

permissions:
  contents: write

concurrency:
  group: nompower-generate
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Generate site
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        run: |
          set -euo pipefail
          if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
            echo "[ERROR] SecretsにDEEPSEEK_API_KEYを入れてください" >&2
            exit 1
          fi
          python nompower_pipeline/generate.py

      - name: Commit & push (if changed)
        run: |
          set -euo pipefail
          git config user.name "nompower-bot"
          git config user.email "nompower-bot@users.noreply.github.com"

          git add -A
          if git diff --cached --quiet; then
            echo "[LOG] No changes."
            exit 0
          fi

          git commit -m "Auto-generate site"
          git push
