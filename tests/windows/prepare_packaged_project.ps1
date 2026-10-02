param([int]$StudioPid)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes
$deadline = [DateTime]::UtcNow.AddSeconds(35)

function Find-Control([string]$suffix) {
    $window = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
        [System.Windows.Automation.TreeScope]::Children,
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $StudioPid
        )
    )
    if ($null -eq $window) { return $null }
    $window.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition
    ) | Where-Object { $_.Current.AutomationId.EndsWith(".$suffix") }
}

foreach ($controlName in @('continueToProjectButton', 'reviewSetupButton', 'prepareOnlyButton')) {
    do {
        $control = Find-Control $controlName
        if ($null -ne $control -and $control.Current.IsEnabled) { break }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    if ($null -eq $control -or -not $control.Current.IsEnabled) {
        throw "Packaged Studio did not enable $controlName"
    }
    $control.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 200
}

do {
    $review = Find-Control 'reviewPreparedButton'
    if ($null -ne $review -and $review.Current.IsEnabled) {
        Write-Output 'Packaged preparation reached the result page.'
        exit 0
    }
    Start-Sleep -Milliseconds 100
} while ([DateTime]::UtcNow -lt $deadline)
throw 'Packaged preparation did not reach the result page.'
