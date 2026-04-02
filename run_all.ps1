# Usage:
#   .\run_all.ps1
#   .\run_all.ps1 -SkipHS
#   .\run_all.ps1 -SeedsOnly 42
#
# Prerequisites:
#   - Python 3.10+ on PATH
#   - GPU recommended
# ═══════════════════════════════════════════════════════════════════

param(
    [int[]]$SeedsOnly,
    [switch]$SkipHS,
    [switch]$SkipPlots,
    [int]$NPerm = 100,
    [int]$MaxEpochs = 50
)

$ErrorActionPreference = "Stop"

$SEEDS = if ($SeedsOnly) { $SeedsOnly } else { @(42, 123, 456, 789, 1024) }
$DATASETS = @("industry12", "industry49")
$PORTFOLIOS = @("equal_weight", "vol_scaled")
$ALPHAS = @("0.05", "0.01")
$MODELS = @("flatten_mlp", "set_attention")

# ── Helpers ───────────────────────────────────────────────────────

function Write-Banner($msg) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step($current, $total, $label, $color = "Yellow") {
    Write-Host "[$current/$total] " -NoNewline -ForegroundColor DarkGray
    Write-Host $label -ForegroundColor $color
}

# ── Step 1: Install dependencies ─────────────────────────────────

Write-Banner "Step 1: Install dependencies"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed" -ForegroundColor Red
    exit 1
}

# ── Step 2: Download Fama-French data ────────────────────────────

Write-Banner "Step 2: Download Fama-French data"

if (-not (Test-Path "data\raw\ff_12_industry_daily.csv") -or
    -not (Test-Path "data\raw\ff_49_industry_daily.csv")) {
    python -m src.data.download_ff_data
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Auto-download failed. Please download manually:" -ForegroundColor Yellow
        Write-Host "  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
        Write-Host "  Place CSVs in data\raw\"
        Read-Host "Press Enter after placing files, or Ctrl+C to abort"
    }
} else {
    Write-Host "Data files already present, skipping download."
}

# ── Step 3: Neural model experiments ─────────────────────────────

Write-Banner "Step 3: Train neural models ($($SEEDS.Count) seeds x 2 datasets x 2 portfolios x 2 alphas x 2 models)"

$configs = @()
foreach ($dataset in $DATASETS) {
    foreach ($portfolio in $PORTFOLIOS) {
        foreach ($alpha in $ALPHAS) {
            foreach ($model in $MODELS) {
                foreach ($seed in $SEEDS) {
                    $configs += [PSCustomObject]@{
                        Dataset   = $dataset
                        Portfolio = $portfolio
                        Alpha     = $alpha
                        Model     = $model
                        Seed      = $seed
                    }
                }
            }
        }
    }
}

$total = $configs.Count
$current = 0
$failed = @()

foreach ($cfg in $configs) {
    $current++
    $runName = "$($cfg.Dataset)__$($cfg.Portfolio)__$($cfg.Model)__alpha_$($cfg.Alpha)__seed_$($cfg.Seed)"
    $outDir = "artifacts\$runName"
    $metricsPath = "$outDir\metrics.json"

    # Skip if already done
    if (Test-Path $metricsPath) {
        Write-Step $current $total "SKIP (exists): $runName" "DarkGray"
        continue
    }

    # ── Training ──
    Write-Step $current $total "TRAINING: $runName" "Yellow"

    python -m src.experiments.run_experiment `
        --dataset $cfg.Dataset `
        --portfolio $cfg.Portfolio `
        --model $cfg.Model `
        --alpha $cfg.Alpha `
        --seed $cfg.Seed `
        --max_epochs $MaxEpochs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED training: $runName" -ForegroundColor Red
        $failed += "train:$runName"
        continue
    }

    # ── Permutation test ──
    $ckptPath = "$outDir\model.pt"
    if (Test-Path $ckptPath) {
        Write-Step $current $total "PERM TEST ($NPerm perms): $runName" "Magenta"

        python -m src.experiments.run_permutation_test `
            --dataset $cfg.Dataset `
            --portfolio $cfg.Portfolio `
            --model $cfg.Model `
            --checkpoint $ckptPath `
            --n_perm $NPerm `
            --seed $cfg.Seed

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED permutation test: $runName" -ForegroundColor Red
            $failed += "perm:$runName"
        }
    }
}

# ── Step 4: Historical Simulation baselines ──────────────────────

if (-not $SkipHS) {
    Write-Banner "Step 4: Historical Simulation baselines"
    python -m src.experiments.run_historical_simulation
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Historical simulation had errors" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping Historical Simulation (flag -SkipHS)" -ForegroundColor DarkGray
}

# ── Step 5: Aggregate results ────────────────────────────────────

Write-Banner "Step 5: Aggregate results"
python -m src.experiments.aggregate_results
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Aggregation failed" -ForegroundColor Red
    exit 1
}

# ── Step 6: Generate figures ─────────────────────────────────────

if (-not $SkipPlots) {
    Write-Banner "Step 6: Generate figures"
    python -m src.experiments.plot_results
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Some plots may have failed" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping plots (flag -SkipPlots)" -ForegroundColor DarkGray
}

# ── Summary ──────────────────────────────────────────────────────

Write-Banner "COMPLETE"

Write-Host "Results:" -ForegroundColor Green
Write-Host "  artifacts\multi_seed_summary.csv"
Write-Host "  artifacts\table_forecasting.tex"
Write-Host "  artifacts\table_permutation.tex"
Write-Host "  paper\figures\*.pdf"

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed runs ($($failed.Count)):" -ForegroundColor Red
    foreach ($f in $failed) {
        Write-Host "  $f" -ForegroundColor Red
    }
}

# Count completed runs
$completedMetrics = (Get-ChildItem -Path "artifacts" -Filter "metrics.json" -Recurse).Count
$completedPerms = (Get-ChildItem -Path "artifacts" -Filter "permutation_test.json" -Recurse).Count

Write-Host ""
Write-Host "Completed: $completedMetrics metric files, $completedPerms permutation tests" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Copy LaTeX tables from artifacts\ into paper\main.tex"
Write-Host "  2. Copy figures from paper\figures\ to your LaTeX build"
Write-Host "  3. Compile: cd paper; pdflatex main; bibtex main; pdflatex main; pdflatex main"
Write-Host "  4. Create GitHub repo and update URL in paper"
