@echo off
echo ========================================================
echo  CS116 Kaggle S6E10 - Pipeline LLR to OOF and Submission
echo ========================================================
echo.

where python | findstr /i "cs116" >nul
if errorlevel 1 (
    echo Kich hoat moi truong conda cs116...
    call "C:\Users\Asus\miniconda3\condabin\conda.bat" activate cs116
)

python run_llr_pipeline.py

echo.
echo Hoan thanh! Kiem tra file nop tai: submissions\sub_cs116_llr_xgb_v1.csv
pause
