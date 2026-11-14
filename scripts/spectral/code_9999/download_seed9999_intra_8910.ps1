# Download ONLY seed9999 INTRA checkpoints for given subjects (default 8 9 10).
# Exact size match to remote Content-Length; single-instance.
param(
    [int[]]$SubIds = @(8, 9, 10)
)

$ErrorActionPreference = "Continue"
$Root = "D:\NeuroBridge\checkpoints\seed9999\intra"
$Repo = "https://media.githubusercontent.com/media/junmoxiao-cloud/2026BMI-/main"
$Log = "D:\NeuroBridge\logs\download_seed9999_intra_8910.log"
$LockFile = "D:\NeuroBridge\tmp\download_seed9999_intra_8910.lock"
$Expected = 77464282

New-Item -ItemType Directory -Force -Path $Root, (Split-Path $Log), "D:\NeuroBridge\tmp" | Out-Null
$env:TEMP = "D:\NeuroBridge\tmp"
$env:TMP = "D:\NeuroBridge\tmp"

if (Test-Path $LockFile) {
    $oldPid = Get-Content $LockFile -ErrorAction SilentlyContinue
    if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-Host "Another intra 8/9/10 download running (PID $oldPid). Exit."
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

$Intra = @{
    8 = "NeuroBridge/9999-intra-subjects/20260728-193502-sub-08/checkpoint_test_best.pth"
    9 = "NeuroBridge/9999-intra-subjects/20260728-193823-sub-09/checkpoint_test_best.pth"
    10 = "NeuroBridge/9999-intra-subjects/20260728-194145-sub-10/checkpoint_test_best.pth"
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
Log "START intra-only download Subs=$($SubIds -join ',') PID=$PID expected=$Expected"

try {
    foreach ($sid in ($SubIds | Sort-Object)) {
        if (-not $Intra.ContainsKey($sid)) {
            Log "SKIP unknown sid $sid"
            continue
        }
        $rel = $Intra[$sid]
        $dir = Join-Path $Root ("sub-{0:D2}-seed9999" -f $sid)
        $dst = Join-Path $dir "checkpoint_test_best.pth"
        $part = "$dst.part"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null

        if (Test-Path $dst) {
            $len = (Get-Item $dst).Length
            if (-not (Is-Good $dst)) {
                Log "REMOVE bad intra sub-$('{0:D2}' -f $sid) size=$len (want $Expected)"
                Remove-Item $dst -Force
            }
        }
        if (Is-Good $dst) {
            Log "SKIP intra sub-$('{0:D2}' -f $sid) ($Expected bytes)"
            $ok++
            continue
        }

        # Drop oversized/undersized partials that can never finish via Range resume
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
                Log "GIVE UP intra sub-$('{0:D2}' -f $sid)"
                $fail++
                break
            }
            Log "GET intra sub-$('{0:D2}' -f $sid) attempt $attempt"
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
