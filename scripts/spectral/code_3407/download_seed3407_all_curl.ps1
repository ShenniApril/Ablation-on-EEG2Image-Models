$ErrorActionPreference = "Stop"
$Base = "https://media.githubusercontent.com/media/junmoxiao-cloud/2026BMI-/main"
$Root = "D:\NeuroBridge\checkpoints\seed3407"
$ExpIntra = 77464705
$ExpInter = 44193613
$Tol = 0.15

function Ok([string]$path, [int64]$exp) {
    if (-not (Test-Path $path)) { return $false }
    $len = (Get-Item $path).Length
    $lo = [int64]($exp * (1 - $Tol))
    $hi = [int64]($exp * (1 + $Tol))
    return ($len -ge $lo -and $len -le $hi)
}

foreach ($mode in @("intra", "inter")) {
    $exp = if ($mode -eq "intra") { $ExpIntra } else { $ExpInter }
    $prefix = if ($mode -eq "intra") { "intra-subjects-seed3407" } else { "inter-subjects-seed3407" }
    for ($sid = 1; $sid -le 10; $sid++) {
        $s = "{0:D2}" -f $sid
        $dst = Join-Path $Root "$mode\sub-$s-seed3407\checkpoint_test_best.pth"
        if (Ok $dst $exp) { Write-Host "SKIP $mode sub-$s"; continue }
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        if (Test-Path $dst) { Remove-Item $dst -Force }
        $url = "$Base/results/$prefix/sub-$s-seed3407/checkpoint_test_best.pth"
        Write-Host "GET $mode sub-$s"
        curl.exe -L --retry 5 --retry-delay 5 -o $dst $url
        if (-not (Ok $dst $exp)) { throw "Bad $mode sub-$s size" }
    }
}
Write-Host "All seed3407 checkpoints OK"
