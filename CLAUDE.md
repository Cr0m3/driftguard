# DriftGuard — Projekt-Kontext

## Was ist DriftGuard
Config-Drift-Detector für OpenTofu/Terraform + Azure.
Vergleicht IaC-Soll-Zustand mit Cloud-Ist-Zustand.

## Tech Stack
- Python 3.11 (Analyzer, Claude SDK)
- Node 20 + Next.js 14 (Dashboard)
- OpenTofu >= 1.6
- Azure CLI + Azure Monitor API

## Wichtige Befehle
- `python3 src/analyzer.py` — lokaler Drift-Scan
- `tofu plan -json > plan.json` — Plan generieren
- `npm run dev` — Dashboard starten

## Coding-Konventionen
- Keine Cloud-Credentials im Code, nur env vars
- Alle Azure-Calls read-only (keine Write-Permissions)
- Jede Funktion hat einen Docstring