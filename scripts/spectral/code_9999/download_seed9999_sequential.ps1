# Sequential seed9999 checkpoint download (curl resume). One file at a time.
$ErrorActionPreference = "Continue"
$Root = "D:\NeuroBridge\checkpoints\seed9999"
$Repo = "https://media.githubusercontent.com/media/junmoxiao-cloud/2026BMI-/main"
$Log = "D:\NeuroBridge\logs\download_seed9999_sequential.log"
$LogMirror = "D:\git\NeuroBridge\data\download_seed9999_status.log"
$LockFile = "D:\NeuroBridge\tmp\download_seed9999.lock"
$ExpectedIntra = 77464282
$ExpectedInter = 44193126
$Tol = 0.15

New-Item -ItemType Directory -Force -Path $Root, (Split-Path $Log), (Split-Path $LogMirror), "D:\NeuroBridge\tmp" | Out-Null
$env:TEMP = "D:\NeuroBridge\tmp"
$env:TMP = "D:\NeuroBridge\tmp"

if (Test-Path $LockFile) {
    $oldPid = Get-Content $LockFile -ErrorAction SilentlyContinue
    if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-Host "Another download is running (PID $oldPid). Exit."
        exit 1
    }
    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}
Set-Content -Path $LockFile -Value $PID -Encoding ascii

function Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    foreach ($path in @($Log, $LogMirror)) {
        try { Add-Content -Path $path -Value $line -Encoding utf8 } catch {}
    }
    Write-Host $line
}

function Expected-Bytes([string]$mode) {
    if ($mode -eq "inter") { return $ExpectedInter }
    return $ExpectedIntra
}

function Is-RealCheckpoint([string]$path, [string]$mode) {
    if (-not (Test-Path $path)) { return $false }
    $len = (Get-Item $path).Length
    $exp = Expected-Bytes $mode
    $lo = [int64]($exp * (1 - $Tol))
    $hi = [int64]($exp * (1 + $Tol))
    if ($len -lt $lo -or $len -gt $hi) { return $false }
    $buf = New-Object byte[] 80
    $fs = [System.IO.File]::OpenRead($path)
    try {
        $null = $fs.Read($buf, 0, 80)
    } finally {
        $fs.Close()
    }
    $text = [System.Text.Encoding]::ASCII.GetString($buf)
    return -not $text.StartsWith("version https://git-lfs.github.com")
}

function Sanitize-Partial([string]$part, [string]$mode) {
    if (-not (Test-Path $part)) { return }
    $exp = Expected-Bytes $mode
    $hi = [int64]($exp * (1 + $Tol))
    $len = (Get-Item $part).Length
    if ($len -eq 0 -or $len -gt $hi) {
        Remove-Item $part -Force -ErrorAction SilentlyContinue
        Log "  removed bad part ($len bytes, max $hi)"
    }
}

$Intra = @{
    1 = "NeuroBridge/9999-intra-subjects/20260728-191248-sub-01/checkpoint_test_best.pth"
    2 = "NeuroBridge/9999-intra-subjects/20260728-191556-sub-02/checkpoint_test_best.pth"
    3 = "NeuroBridge/9999-intra-subjects/20260728-191913-sub-03/checkpoint_test_best.pth"
    4 = "NeuroBridge/9999-intra-subjects/20260728-192218-sub-04/checkpoint_test_best.pth"
    5 = "NeuroBridge/9999-intra-subjects/20260728-192522-sub-05/checkpoint_test_best.pth"
    6 = "NeuroBridge/9999-intra-subjects/20260728-192834-sub-06/checkpoint_test_best.pth"
    7 = "NeuroBridge/9999-intra-subjects/20260728-193145-sub-07/checkpoint_test_best.pth"
    8 = "NeuroBridge/9999-intra-subjects/20260728-193502-sub-08/checkpoint_test_best.pth"
    9 = "NeuroBridge/9999-intra-subjects/20260728-193823-sub-09/checkpoint_test_best.pth"
    10 = "NeuroBridge/9999-intra-subjects/20260728-194145-sub-10/checkpoint_test_best.pth"
}
$Inter = @{
    1 = "NeuroBridge/9999-inter-subjects/20260728-200756-sub-01/checkpoint_test_best.pth"
    2 = "NeuroBridge/9999-inter-subjects/20260728-214203-sub-02/checkpoint_test_best.pth"
    3 = "NeuroBridge/9999-inter-subjects/20260728-231748-sub-03/checkpoint_test_best.pth"
    4 = "NeuroBridge/9999-inter-subjects/20260729-005036-sub-04/checkpoint_test_best.pth"
    5 = "NeuroBridge/9999-inter-subjects/20260729-022034-sub-05/checkpoint_test_best.pth"
    6 = "NeuroBridge/9999-inter-subjects/20260729-035153-sub-06/checkpoint_test_best.pth"
    7 = "NeuroBridge/9999-inter-subjects/20260729-052319-sub-07/checkpoint_test_best.pth"
    8 = "NeuroBridge/9999-inter-subjects/20260729-065345-sub-08/checkpoint_test_best.pth"
    9 = "NeuroBridge/9999-inter-subjects/20260729-082505-sub-09/checkpoint_test_best.pth"
    10 = "NeuroBridge/9999-inter-subjects/20260729-095434-sub-10/checkpoint_test_best.pth"
}

$ok = 0
$fail = 0
Log "START sequential seed9999 download (PID $PID)"

try {
    foreach ($mode in @("intra", "inter")) {
        $map = if ($mode -eq "intra") { $Intra } else { $Inter }
        foreach ($sid in ($map.Keys | Sort-Object)) {
            $rel = $map[$sid]
            $dir = Join-Path $Root "$mode\sub-$('{0:D2}' -f $sid)-seed9999"
            $dst = Join-Path $dir "checkpoint_test_best.pth"
            $part = "$dst.part"
            New-Item -ItemType Directory -Force -Path $dir | Out-Null

            if (Test-Path $dst) {
                if (-not (Is-RealCheckpoint $dst $mode)) {
                    Log "REMOVE bad $mode sub-$('{0:D2}' -f $sid) ($([math]::Round((Get-Item $dst).Length/1MB,1)) MB)"
                    Remove-Item $dst -Force
                }
            }

            if (Is-RealCheckpoint $dst $mode) {
                Log "SKIP $mode sub-$('{0:D2}' -f $sid) ($([math]::Round((Get-Item $dst).Length/1MB,1)) MB)"
                $ok++
                continue
            }

            Sanitize-Partial $part $mode
            $url = "$Repo/$($rel -replace '\\','/')"
            $curl = "C:\Windows\System32\curl.exe"
            $attempt = 0
            $maxAttempts = 50
            while (-not (Is-RealCheckpoint $dst $mode)) {
                $attempt++
                if ($attempt -gt $maxAttempts) {
                    Log "GIVE UP $mode sub-$('{0:D2}' -f $sid) after $maxAttempts attempts"
                    $fail++
                    break
                }
                Log "GET $mode sub-$('{0:D2}' -f $sid) attempt $attempt"
                if ((Test-Path $part) -and (Get-Item $part).Length -gt 0) {
                    Log "  resume part $([math]::Round((Get-Item $part).Length/1MB,2)) MB"
                }
                & $curl -L --retry 5 --retry-delay 10 --connect-timeout 120 --max-time 7200 -C - -o $part $url
                if ($LASTEXITCODE -ne 0) {
                    Log "  FAIL curl exit $LASTEXITCODE, retry in 30s"
                    Start-Sleep -Seconds 30
                    continue
                }
                if (-not (Test-Path $part)) {
                    Log "  FAIL missing part, retry in 30s"
                    Start-Sleep -Seconds 30
                    continue
                }
                $plen = (Get-Item $part).Length
                $exp = Expected-Bytes $mode
                $lo = [int64]($exp * (1 - $Tol))
                $hi = [int64]($exp * (1 + $Tol))
                if ($plen -lt $lo -or $plen -gt $hi) {
                    Log "  FAIL part size $plen (want $lo..$hi), retry in 30s"
                    if ($plen -gt $hi) { Remove-Item $part -Force -ErrorAction SilentlyContinue }
                    Start-Sleep -Seconds 30
                    continue
                }
                Move-Item -Force $part $dst
                Log "  OK $dst ($([math]::Round((Get-Item $dst).Length/1MB,1)) MB)"
            }
            if (Is-RealCheckpoint $dst $mode) { $ok++ }
        }
    }
    Log "DONE ok=$ok fail=$fail"
}
finally {
    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}
