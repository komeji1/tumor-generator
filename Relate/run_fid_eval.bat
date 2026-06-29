@echo off
REM === FID Evaluation Runner ===
REM Two evaluations:
REM   1. autoPET50 vs MAISI step1 (paper benchmark)
REM   2. MAISI step1 vs MAISI step2 (bridge before/after)

cd /d "%~dp0.."

echo ============================================================
echo  FID Evaluation 1: autoPET (real) vs MAISI step1 (synth)
echo  Purpose: Compare with MAISI paper FID = 5.124
echo ============================================================
echo.

python Relate\compute_fid_single_gpu.py ^
    --real_dataset_root "data/autopet_ct50" ^
    --real_filelist "data/autopet_ct50/filelist.txt" ^
    --real_features_dir "autopet_real" ^
    --synth_dataset_root "output" ^
    --synth_filelist "data/fid_maisi_step1/filelist.txt" ^
    --synth_features_dir "maisi_step1" ^
    --model_name "radimagenet_resnet50" ^
    --target_shape "256x256x128" ^
    --enable_resampling_spacing "1.0x1.0x1.0" ^
    --enable_center_slices_ratio 0.4 ^
    --enable_padding True ^
    --enable_center_cropping True ^
    --ignore_existing False ^
    --num_images 50 ^
    --output_root "data/fid_features"

echo.
echo ============================================================
echo  FID Evaluation 2: MAISI step1 (real) vs step2 (synth)
echo  Purpose: Measure quality change after adding tumors
echo ============================================================
echo.

python Relate\compute_fid_single_gpu.py ^
    --real_dataset_root "output" ^
    --real_filelist "data/fid_maisi_step1/filelist.txt" ^
    --real_features_dir "maisi_step1" ^
    --synth_dataset_root "output" ^
    --synth_filelist "data/fid_maisi_step2/filelist.txt" ^
    --synth_features_dir "maisi_step2" ^
    --model_name "radimagenet_resnet50" ^
    --target_shape "256x256x128" ^
    --enable_resampling_spacing "1.0x1.0x1.0" ^
    --enable_center_slices_ratio 0.4 ^
    --enable_padding True ^
    --enable_center_cropping True ^
    --ignore_existing False ^
    --num_images 8 ^
    --output_root "data/fid_features"

echo.
echo Done.
pause
