$procs = Get-CimInstance Win32_Process -Filter "name='python.exe'"
foreach ($p in $procs) {
  if ($p.CommandLine -like '*watch.py*') {
    Write-Output ("PID=" + $p.ProcessId + " CREATED=" + $p.CreationDate)
  }
}
Write-Output "scan done"
