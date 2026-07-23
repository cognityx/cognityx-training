param(
    [string]$OutputPath = "D:\AI\models\cognityx\telemetry\windows-host.json",
    [double]$DedicatedTotalGiB = 31.5,
    [double]$InstalledMemoryGiB = 128,
    [double]$IntervalSeconds = 1
)

$ErrorActionPreference = "Stop"
$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$temporaryPath = "$OutputPath.tmp"
$counterPaths = @(
    "\GPU Adapter Memory(*)\Dedicated Usage",
    "\GPU Adapter Memory(*)\Shared Usage"
)

function ConvertTo-NonnegativeInt64 {
    param([double]$Value)

    if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) {
        throw "GPU memory performance counter returned a non-finite value: $Value"
    }
    return [int64][Math]::Max([double]0, [double]$Value)
}

Write-Host "Writing Windows GPU/host telemetry to $OutputPath"
Write-Host "Press Ctrl+C to stop."

while ($true) {
    $counter = Get-Counter -Counter $counterPaths -ErrorAction Stop
    $dedicatedUsed = [double](
        $counter.CounterSamples |
        Where-Object { $_.Path -like "*dedicated usage" } |
        Measure-Object -Property CookedValue -Sum
    ).Sum
    $sharedUsed = [double](
        $counter.CounterSamples |
        Where-Object { $_.Path -like "*shared usage" } |
        Measure-Object -Property CookedValue -Sum
    ).Sum
    $computer = Get-CimInstance Win32_ComputerSystem
    $operatingSystem = Get-CimInstance Win32_OperatingSystem
    $processor = Get-CimInstance Win32_Processor |
        Measure-Object -Property LoadPercentage -Average
    $installedBytes = [int64]$computer.TotalPhysicalMemory
    $availableBytes = [int64]$operatingSystem.FreePhysicalMemory * 1024
    $dedicatedTotalBytes = [int64]($DedicatedTotalGiB * 1GB)
    $sharedTotalBytes = [int64](($InstalledMemoryGiB / 2) * 1GB)

    $sample = [ordered]@{
        schema_version = "1.0"
        captured_at_utc = [DateTime]::UtcNow.ToString("o")
        source = "windows_performance_counters"
        dedicated_used_bytes = ConvertTo-NonnegativeInt64 $dedicatedUsed
        dedicated_total_bytes = $dedicatedTotalBytes
        shared_used_bytes = ConvertTo-NonnegativeInt64 $sharedUsed
        shared_total_bytes = $sharedTotalBytes
        combined_used_bytes = ConvertTo-NonnegativeInt64 (
            [double]$dedicatedUsed + [double]$sharedUsed
        )
        combined_total_bytes = $dedicatedTotalBytes + $sharedTotalBytes
        host_memory_used_bytes = $installedBytes - $availableBytes
        host_memory_total_bytes = $installedBytes
        host_cpu_percent = [double]$processor.Average
    }
    $sample | ConvertTo-Json -Compress | Set-Content -Encoding UTF8 $temporaryPath
    Move-Item -Force $temporaryPath $OutputPath
    Start-Sleep -Milliseconds ([int]($IntervalSeconds * 1000))
}
