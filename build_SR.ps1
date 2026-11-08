# Builds the Scientific Reports submission: main_SR.pdf (article) and
# supplementary_SR.pdf (Supplementary Information), which cite each other's
# labels through xr-hyper and therefore have to be compiled alternately.
# Usage:  .\build_SR.ps1            (from paper1_medical/)
#         .\build_SR.ps1 -Clean     (discard auxiliary files first)
param([switch]$Clean)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:BIBINPUTS = "..;.;"

if ($Clean) {
    latexmk -C main_SR.tex | Out-Null
    latexmk -C supplementary_SR.tex | Out-Null
}

# Two rounds are enough: round 1 creates both .aux files, round 2 resolves
# every cross-document reference on both sides.
foreach ($round in 1, 2) {
    Write-Host "=== round ${round}: article ==="
    latexmk -pdf -interaction=nonstopmode -halt-on-error main_SR.tex
    if ($LASTEXITCODE -ne 0) { throw "main_SR.tex failed (round $round)" }
    Write-Host "=== round ${round}: supplementary ==="
    latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_SR.tex
    if ($LASTEXITCODE -ne 0) { throw "supplementary_SR.tex failed (round $round)" }
}

# Label snapshots for Overleaf, which strips .aux files from uploaded ZIPs.
python export_xr_labels.py

Write-Host ""
Write-Host "=== warnings that matter ==="
foreach ($log in "main_SR.log", "supplementary_SR.log") {
    $hits = Select-String -Path $log -Pattern "undefined|multiply defined|Float too large|Overfull \\hbox \((\d{3,})" |
        Where-Object { $_.Line -notmatch "^Package: " }
    Write-Host ("{0}: {1} line(s)" -f $log, @($hits).Count)
    $hits | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
}
python wordcount_sr.py
