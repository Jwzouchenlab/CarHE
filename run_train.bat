@echo off
REM ============================================================
REM CarHE Xenium Training — 一键运行
REM ============================================================
REM 在 CarHE/ 目录下运行此脚本
REM ============================================================

echo ============================================================
echo CarHE: Xenium Data -^> Training Pipeline
echo ============================================================

REM --- Step 1: Convert Xenium to h5ad ---
echo.
echo [Step 1/3] Converting Xenium data to AnnData...
cd data
python quick_xenium_to_h5ad.py
cd ..
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conversion failed!
    pause
    exit /b 1
)

REM --- Step 2: Train ---
echo.
echo [Step 2/3] Training CarHE model...
cd model
python train.py --adata ../data/xenium_prostate.h5ad --batch_size 16 --epochs 30 --exp_name ./checkpoint/xenium_model
cd ..
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

REM --- Step 3: Evaluate ---
echo.
echo [Step 3/3] Evaluating model...
cd model
python evaluate.py --adata ../data/xenium_prostate.h5ad --checkpoint ./checkpoint/xenium_model/best_model.pt --output ./evaluation_results/xenium
cd ..

echo.
echo ============================================================
echo Pipeline complete! Results:
echo   Model: model/checkpoint/xenium_model/best_model.pt
echo   Eval:  model/evaluation_results/xenium/
echo ============================================================
pause
