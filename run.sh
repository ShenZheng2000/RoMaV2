# NOTE: --img_A => nearview. --img_B => farview.

# # # default imagees 
# python demo/demo_covariance.py

# # # from khiem's aerial image! 
# python demo/demo_covariance.py \
#     --im_A_path assets/t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10/00572.jpg \
#     --im_B_path assets/t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10/00563.jpg \
#     --save_path demo/output/00572_vs_00563/roma_v2_std.png

# python demo/demo_covariance.py \
#     --im_A_path assets/t03_v07_s00_r01_ReconstructedArea_WACV_test_A09/00412.jpg \
#     --im_B_path assets/t03_v07_s00_r01_ReconstructedArea_WACV_test_A09/00399.jpg \
#     --save_path demo/output/00399_vs_00412/roma_v2_std.png

# CUDA_VISIBLE_DEVICES=7 python demo/demo_covariance.py \
#     --im_A_path assets/t04_v07_s02_r02_VaryingAltitudes_M07_building_1_door/image_000020.jpg \
#     --im_B_path assets/t04_v07_s02_r02_VaryingAltitudes_M07_building_1_door/image_000003.jpg \
#     --save_path demo/output/00003_vs_00020/roma_v2_std.png


for fwhm in 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    CUDA_VISIBLE_DEVICES=7 python demo/demo_covariance.py \
        --im_A_path assets/t04_v07_s02_r02_VaryingAltitudes_M07_building_1_door/image_000020.jpg \
        --im_B_path assets/t04_v07_s02_r02_VaryingAltitudes_M07_building_1_door/image_000003.jpg \
        --save_path demo/output/00003_vs_00020/roma_v2_std.png \
        --attraction_fwhm $fwhm
done


# for testing fps ONLY
# python tests/test_fps.py