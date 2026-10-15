# Runs after the hand-held fine-tune ends: detector selection + one test read, then the voice design evaluation.
# One heavy job family at a time (paper_evidence/CRASH_FIXED.md).
$py = "C:\Users\LAM\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$env:PYTHONIOENCODING = "utf-8"
while (Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'train_closeups' }) { Start-Sleep 60 }
Remove-Item Env:EVAL_DEVICE -ErrorAction SilentlyContinue
$det = "E:\Final SP\realtime_bangla_taka_detection"
Set-Location $det
& $py -u "$det\scripts\eval\eval_closeup_ft.py" select *> "E:\Final SP\closeup_select2.log"
& $py -u "$det\scripts\eval\eval_closeup_ft.py" test *> "E:\Final SP\closeup_test.log"
Set-Location "E:\Final SP\savior_glass"
$procs = foreach ($k in 0..3) {
  Start-Process -FilePath $py -ArgumentList "-u","scripts/voice_eval.py","design","$k","4" -WorkingDirectory "E:\Final SP\savior_glass" -RedirectStandardOutput "E:\Final SP\voice_eval_$k.log" -RedirectStandardError "E:\Final SP\voice_eval_$k.err" -WindowStyle Hidden -PassThru
}
$procs | Wait-Process
& $py -u scripts/voice_eval.py design *> "E:\Final SP\voice_eval_choose.log"
& $py -u scripts/voice_eval.py test *> "E:\Final SP\voice_eval_test.log"
"all done" | Out-File "E:\Final SP\after_training.done"
