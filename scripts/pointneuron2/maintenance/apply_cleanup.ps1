param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$originalRoot = [IO.Directory]::GetParent($workspaceRoot).FullName
$generatedRoot = Join-Path $workspaceRoot 'tmp\pointneuron2'
$planPath = Join-Path $generatedRoot 'preserved_reference\cleanup_plan.json'
$planData = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($planData.workspace -ne $workspaceRoot -or $planData.original -ne $originalRoot) { throw 'Wrong workspace' }
if ($planData.status -ne 'archive_verified_pending_cleanup') { throw 'Plan is not pending' }
$archivePath = [IO.Path]::GetFullPath($planData.archive)
if (-not $archivePath.StartsWith((Join-Path $generatedRoot 'preserved_reference\'), [StringComparison]::OrdinalIgnoreCase)) { throw 'Archive escapes preservation directory' }
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $planData.archive_sha256) { throw 'Archive hash mismatch' }
$scratchNames = @('pointneuron2_audit','pointneuron2_layout','pointneuron2_sampling','pointneuron2_staging','pointneuron2_training','pointneuron2_workspace_backup','pointneuron2_review_ui_check.js','pointneuron2_workspace_transfer.json','pointneuron2_workspace_transfer.ps1')
$validated = @()
$linksToUnlink = @()
foreach ($target in $planData.targets) {
    $absolute = [IO.Path]::GetFullPath($target.path)
    if (-not (Test-Path -LiteralPath $absolute)) { continue } # Safe replay after a partially completed cleanup.
    $item = Get-Item -LiteralPath $absolute -Force
    if ($item.FullName -ne $absolute -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Invalid top-level target: $absolute" }
    $parent = [IO.Directory]::GetParent($absolute).FullName
    $allowed = ($parent -eq $generatedRoot -and $item.Name -notin @('sampling_env','pointnext_reference','preserved_reference'))
    $allowed = $allowed -or ($absolute -eq (Join-Path $workspaceRoot 'tmp\pdfs')) -or ($absolute -eq (Join-Path $workspaceRoot '.pytest_cache'))
    $allowed = $allowed -or ($parent -eq (Join-Path $originalRoot 'tmp') -and $item.Name -in $scratchNames)
    $allowed = $allowed -or ($parent -eq (Join-Path $workspaceRoot 'output\pdf') -and $item.Extension -in @('.aux','.log','.out'))
    foreach ($sourceFolder in @('src','scripts','tests')) {
        $prefix = Join-Path $workspaceRoot ($sourceFolder + '\')
        if ($absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -and $item.Name -eq '__pycache__') { $allowed = $true }
    }
    if (-not $allowed) { throw "Target outside approved cleanup scopes: $absolute" }
    # Fresh traversal never follows reparse entries; unlink them separately,
    # without recursion, before recursive removal of ordinary directories.
    $pendingDirectories = [Collections.Generic.Stack[string]]::new()
    if ($item.PSIsContainer) { $pendingDirectories.Push($absolute) }
    while ($pendingDirectories.Count -gt 0) {
        foreach ($child in (Get-ChildItem -LiteralPath $pendingDirectories.Pop() -Force)) {
            if (-not $child.FullName.StartsWith($absolute + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Child outside target' }
            if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) { $linksToUnlink += $child }
            elseif ($child.PSIsContainer) { $pendingDirectories.Push($child.FullName) }
        }
    }
    foreach ($linkItem in $linksToUnlink) {
        foreach ($destination in $linkItem.Target) {
            $resolved = if ([IO.Path]::IsPathRooted($destination)) { [IO.Path]::GetFullPath($destination) } else { [IO.Path]::GetFullPath((Join-Path $linkItem.Parent.FullName $destination)) }
            if (-not $resolved.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Link escapes workspace: $($linkItem.FullName)" }
        }
    }
    $validated += $absolute
}
foreach ($property in $planData.protected_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) { throw "Protected file changed: $($property.Name)" }
}
Write-Output "Validated $($validated.Count) exact targets; $($planData.target_files) files, $($planData.target_bytes) logical bytes. Archive verified."
if (-not $Apply) { Write-Output 'Preview only; no files deleted.'; return }
foreach ($linkItem in $linksToUnlink) { Remove-Item -LiteralPath $linkItem.FullName -Force }
foreach ($absolute in $validated) {
    Remove-Item -LiteralPath $absolute -Recurse -Force
    if (Test-Path -LiteralPath $absolute) { throw "Deletion incomplete: $absolute" }
}
$planData.status = 'completed'
$planData | Add-Member -NotePropertyName removed_target_count -NotePropertyValue $validated.Count
$planData | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $planPath -Encoding utf8
Write-Output "COMPLETED: removed $($validated.Count) retired output/scratch targets. Kept environment, reference sources and verified evidence archive."
