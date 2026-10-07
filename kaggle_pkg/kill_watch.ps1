$procs = Get-CimInstance Win32_Process -Filter "name='python.exe'"
foreach ($p in $procs) {
  if ($p.CommandLine -like '*watch.py*' -and $p.CommandLine -notlike '*kill*') {
    Write-Output ("killing " + $p.ProcessId + " " + $p.CommandLine)
    Stop-Process -Id $p.ProcessId -Force
  }
}
Write-Output "done"
