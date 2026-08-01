# Registers the KalshiBTCIntelBot scheduled task (runs bot.py at logon,
# hidden via pythonw, no time limit, auto-restart on failure), then starts it.
# Run:  right-click -> Run with PowerShell,  or from a terminal:
#   powershell -ExecutionPolicy Bypass -File install_task.ps1
# To remove later:
#   Unregister-ScheduledTask -TaskName KalshiBTCIntelBot -Confirm:$false

$ErrorActionPreference = "Stop"
$botDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyw = "$env:LOCALAPPDATA\Programs\Python\Python312\pythonw.exe"

# Stop any already-running copy of the bot so the task doesn't duplicate alerts
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "bot\.py" } |
    ForEach-Object {
        Write-Host "Stopping existing bot process $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force
    }

$action = New-ScheduledTaskAction -Execute $pyw -Argument "bot.py" -WorkingDirectory $botDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "KalshiBTCIntelBot" -Action $action `
    -Trigger $trigger -Settings $settings -Force `
    -Description "Kalshi 15-min BTC intel Telegram alert bot (starts at logon)" | Out-Null

Start-ScheduledTask -TaskName "KalshiBTCIntelBot"
Start-Sleep -Seconds 5
$state = (Get-ScheduledTask -TaskName "KalshiBTCIntelBot").State
Write-Host "Task registered. State: $state"
Write-Host "Bot log: $botDir\bot.log"
