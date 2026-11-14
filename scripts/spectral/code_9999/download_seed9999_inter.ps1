# Download seed9999 INTER checkpoints only. Exact size match.
param(
    [int[]]$SubIds = @(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
)

$ErrorActionPreference = "Continue"
$Root = "D:\NeuroBridge\checkpoints\seed9999\inter"
$Repo = "https://media.githubusercontent.com/media/junmoxiao-cloud/2026BMI-/main"
$Log = "D:\NeuroBridge\logs\download_seed9999_inter.log"
$LockFile = "D:\NeuroBridge\tmp\download_seed9999_inter.lock"
$Expected = 44193126

New-Item -ItemType Directory -Force -Path $Root, (Split-Path $Log), "D:\NeuroBridge\tmp" | Out-Null
$env:TEMP = "D:\NeuroBridge\tmp"
$env:TMP = "D:\NeuroBridge\tmp"

if (Test-Path $LockFile) {
    $oldPid = Get-Content $LockFile -ErrorAction SilentlyContinue
    if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-Host "Another inter download running (PID $oldPid). Exit."
        exit 1
    }
    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}
Set-Content -Path $LockFile -Value $PID -Encoding ascii

function Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    try { Add-Content -Path $Log -Value $line -Encoding utf8 } catch {}
    Write-Host $line
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

function Is-Good([string]$path) {
    if (-not (Test-Path $path)) { return $false }
    if ((Get-Item $path).Length -ne $Expected) { return $false }
    $buf = New-Object byte[] 80
    $fs = [System.IO.File]::OpenRead($path)
    try { $null = $fs.Read($buf, 0, 80) } finally { $fs.Close() }
    $text = [System.Text.Encoding]::ASCII.GetString($buf)
    return -not $text.StartsWith("version https://git-lfs.github.com")
}

$ok = 0
$fail = 0
Log "START inter-only download Subs=$($SubIds -join ',') PID=$PID expected=$Expected"

try {
    foreach ($sid in ($SubIds | Sort-Object)) {
        if (-not $Inter.ContainsKey($sid)) { continue }
        $rel = $Inter[$sid]
        $dir = Join-Path $Root ("sub-{0:D2}-seed9999" -f $sid)
        $dst = Join-Path $dir "checkpoint_test_best.pth"
        $part = "$dst.part"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null

        if (Test-Path $dst) {
            $len = (Get-Item $dst).Length
            if (-not (Is-Good $dst)) {
                Log "REMOVE bad inter sub-$('{0:D2}' -f $sid) size=$len (want $Expected)"
                Remove-Item $dst -Force
            }
        }
        if (Is-Good $dst) {
            Log "SKIP inter sub-$('{0:D2}' -f $sid) ($Expected bytes)"
            $ok++
            continue
        }

        if (Test-Path $part) {
            $plen = (Get-Item $part).Length
            if ($plen -eq 0 -or $plen -gt $Expected) {
                Log "  remove bad part ($plen bytes)"
                Remove-Item $part -Force -ErrorAction SilentlyContinue
            }
        }

        $url = "$Repo/$rel"
        $curl = "C:\Windows\System32\curl.exe"
        $attempt = 0
        $maxAttempts = 80
        while (-not (Is-Good $dst)) {
            $attempt++
            if ($attempt -gt $maxAttempts) {
                Log "GIVE UP inter sub-$('{0:D2}' -f $sid)"
                $fail++
                break
            }
            Log "GET inter sub-$('{0:D2}' -f $sid) attempt $attempt"
            if ((Test-Path $part) -and ((Get-Item $part).Length -gt 0)) {
                Log "  resume part $((Get-Item $part).Length) bytes"
            }
            & $curl -L --fail --retry 8 --retry-delay 8 --connect-timeout 120 --max-time 7200 -C - -o $part $url
            $ec = $LASTEXITCODE
            if ($ec -ne 0) {
                Log "  FAIL curl exit $ec, sleep 20s"
                Start-Sleep -Seconds 20
                continue
            }
            if (-not (Test-Path $part)) {
                Log "  FAIL missing part, sleep 20s"
                Start-Sleep -Seconds 20
                continue
            }
            $plen = (Get-Item $part).Length
            if ($plen -ne $Expected) {
                Log "  FAIL size $plen != $Expected"
                if ($plen -gt $Expected) {
                    Remove-Item $part -Force -ErrorAction SilentlyContinue
                }
                Start-Sleep -Seconds 10
                continue
            }
            Move-Item -Force $part $dst
            if (Is-Good $dst) {
                Log "  OK $dst ($Expected bytes)"
            } else {
                Log "  FAIL post-move validation"
                Remove-Item $dst -Force -ErrorAction SilentlyContinue
            }
        }
        if (Is-Good $dst) { $ok++ }
    }
    Log "DONE ok=$ok fail=$fail"
}
finally {
    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}
