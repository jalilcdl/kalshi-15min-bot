# Registers the KalshiPaperTrader scheduled task (runs paper_trader.py at
# logon, hidden via pythonw, no time limit, auto-restart on failure), then
# starts it. Mirrors install_task.ps1 (the alert bot's own task) exactly.
# Run:  right-click -> Run with PowerShell,  or from a terminal:
#   powershell -ExecutionPolicy Bypass -File install_paper_trader_task.ps1
# To remove later:
#   Unregister-ScheduledTask -TaskName KalshiPaperTrader -Confirm:$false
# To check on it any time:
#   Get-ScheduledTask -TaskName KalshiPaperTrader | Get-ScheduledTaskInfo
#   Get-Content paper_trader.log -Tail 40 -Wait

$ErrorActionPreference = "Stop"
$botDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyw = "$env:LOCALAPPDATA\Programs\Python\Python312\pythonw.exe"

if (-not (Test-Path "$botDir\model\strike_prob_model.pkl")) {
    Write-Error "model\strike_prob_model.pkl not found -- run research\strike_probability\scripts\fit_final_model.py first."
}

# Stop any already-running copy so the task doesn't duplicate entries
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "paper_trader\.py" } |
    ForEach-Object {
        Write-Host "Stopping existing paper_trader process $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force
    }

$action = New-ScheduledTaskAction -Execute $pyw -Argument "paper_trader.py" -WorkingDirectory $botDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "KalshiPaperTrader" -Action $action `
    -Trigger $trigger -Settings $settings -Force `
    -Description "Kalshi 15-min BTC strike-probability model: paper trading vs live quotes (starts at logon)" | Out-Null

Start-ScheduledTask -TaskName "KalshiPaperTrader"
Start-Sleep -Seconds 5
$state = (Get-ScheduledTask -TaskName "KalshiPaperTrader").State
Write-Host "Task registered. State: $state"
Write-Host "Paper trader log: $botDir\paper_trader.log"
Write-Host "Trades accumulate in: $botDir\data\trade_log.csv (view them in the dashboard's Trade log page)"
