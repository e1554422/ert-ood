# P0 · 在 tmux 中提取 ViT-B/16 AugReg 的逐层 final_LN(CLS) 特征 (训练子集 100k + val 50k)
$env:PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.JS;.PS1"
$env:PYTHONUTF8 = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:HF_HOME = "D:\CHANGJIANG\hf_cache"
$Py = "C:\Users\wangl4\Miniconda3\envs\terramind\python.exe"
$Banded = "D:\CHANGJIANG\banded_depth_ood"
$Ert = "D:\CHANGJIANG\ert_ood"
$Root = "$Banded\data\datasets\ImageNet1K-trajectory"
$Log = "$Ert\runs\logs"
New-Item -ItemType Directory -Force -Path $Log | Out-Null
Set-Location $Banded
foreach ($Name in @("val50k", "train100k")) {
    & $Py -u scripts\extract_vit_features.py --manifest "$Ert\runs\manifests\$Name.txt" --root $Root `
        --output "$Ert\runs\features\vit_augreg\$Name" --dataset-id "ImageNet1K-$Name" `
        --batch-size 128 --workers 0 --loader-threads 8 --dtype float16 --save-logits *> "$Log\extract_$Name.log"
    "exit $LASTEXITCODE" | Out-File -Append "$Log\extract_$Name.log"
}
"ALL DONE" | Out-File "$Log\extract_done.flag"
