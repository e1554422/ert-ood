# 等特征提取完成后依次跑 P0 (基线复现, 闸门 G0) 与 P1 (逐层 EDL 头, 闸门 G1)
$env:PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.JS;.PS1"
$env:PYTHONUTF8 = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Py = "C:\Users\wangl4\Miniconda3\envs\terramind\python.exe"
$Ert = "D:\CHANGJIANG\ert_ood"
$Log = "$Ert\runs\logs"
while (-not (Test-Path "$Log\extract_done.flag")) { Start-Sleep -Seconds 60 }
Set-Location $Ert
& $Py -u scripts\exp_20261012_p0_baselines_reproduction.py *> "$Log\p0_baselines.log"
"exit $LASTEXITCODE" | Out-File -Append "$Log\p0_baselines.log"
& $Py -u scripts\exp_20261012_p1_edl_heads.py *> "$Log\p1_edl_heads.log"
"exit $LASTEXITCODE" | Out-File -Append "$Log\p1_edl_heads.log"
"DONE" | Out-File "$Log\p0_p1_done.flag"
