$ErrorActionPreference = "Continue"
$Base = "https://media.githubusercontent.com/media/junmoxiao-cloud/2026BMI-/main"
$Root = "D:\NeuroBridge\checkpoints\seed2025\inter"
$Expected = 44193126
$Tol = 0.15
$Inter = @{
    1 = "NeuroBridge/2025-inter-subjects/20260726-185118-sub-01/checkpoint_test_best.pth"
    2 = "NeuroBridge/2025-inter-subjects/20260726-203308-sub-02/checkpoint_test_best.pth"
    3 = "NeuroBridge/2025-inter-subjects/20260726-221709-sub-03/checkpoint_test_best.pth"
    4 = "NeuroBridge/2025-inter-subjects/20260727-000044-sub-04/checkpoint_test_best.pth"
    5 = "NeuroBridge/2025-inter-subjects/20260727-014404-sub-05/checkpoint_test_best.pth"
    6 = "NeuroBridge/2025-inter-subjects/20260727-032756-sub-06/checkpoint_test_best.pth"
    7 = "NeuroBridge/2025-inter-subjects/20260727-051147-sub-07/checkpoint_test_best.pth"
    8 = "NeuroBridge/2025-inter-subjects/20260727-065640-sub-08/checkpoint_test_best.pth"
    9 = "NeuroBridge/2025-inter-subjects/20260727-084155-sub-09/checkpoint_test_best.pth"
    10 = "NeuroBridge/2025-inter-subjects/20260727-102918-sub-10/checkpoint_test_best.pth"
}

function Ok([string]$path) {
    if (-not (Test-Path $path)) { return $false }
    $len = (Get-Item $path).Length
    $lo = [int64]($Expected * (1 - $Tol))
    $hi = [int64]($Expected * (1 + $Tol))
    return ($len -ge $lo -and $len -le $hi)
}

foreach ($sid in 1..10) {
    $s = "{0:D2}" -f $sid
    $dst = Join-Path $Root "sub-$s-seed2025\checkpoint_test_best.pth"
    if (Ok $dst) { Write-Host "SKIP inter sub-$s"; continue }
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    if (Test-Path $dst) { Remove-Item $dst -Force }
    $url = "$Base/$($Inter[$sid])"
    Write-Host "GET inter sub-$s"
    curl.exe -L --retry 5 --retry-delay 5 -o $dst $url
    if (-not (Ok $dst)) { Write-Error "Bad inter sub-$s size"; exit 2 }
}
Write-Host "All inter OK"
