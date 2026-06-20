Add-Type -Assembly System.IO.Compression.FileSystem

$tmpDir = Join-Path $env:TEMP 'chanzi_merge'
if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }

[System.IO.Compression.ZipFile]::ExtractToDirectory('e:\jcc_class\chanzi_test02.zip', $tmpDir)

$files = Get-ChildItem (Join-Path $tmpDir 'chanzi\*.png') | Sort-Object { [int]($_.BaseName) }
$startNum = 43

$dest = [System.IO.Compression.ZipFile]::Open('e:\jcc_class\chanzi_test.zip', 'Update')

foreach ($f in $files) {
    $newName = 'chanzi/' + $startNum + '.png'
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($dest, $f.FullName, $newName) | Out-Null
    Write-Host "Added $newName (from $($f.Name))"
    $startNum++
}

$dest.Dispose()
Remove-Item $tmpDir -Recurse -Force
Write-Host "`nDone. Added files 43-$($startNum - 1) to chanzi_test.zip"
