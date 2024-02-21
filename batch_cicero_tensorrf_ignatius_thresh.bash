export scene=ignatius
export dsf=2
# gardne with cicero_tensorrf 6
python3 warping_evaluation_thresh.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_all/imgs_test_all \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 16 --angle_threshold 4&


# gardne with cicero_tensorrf 6
python3 warping_evaluation_thresh.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_all/imgs_test_all \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 16 --angle_threshold 8&


# gardne with cicero_tensorrf 6
python3 warping_evaluation_thresh.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_all/imgs_test_all \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 16 --angle_threshold 12&


# gardne with cicero_tensorrf 6
python3 warping_evaluation_thresh.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_all/imgs_test_all \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 16 --angle_threshold 16&

# gardne with cicero_tensorrf 6
python3 warping_evaluation_thresh.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_all/imgs_test_all \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 16 --angle_threshold 20&

# gardne with cicero_tensorrf 6
python3 warping_evaluation_thresh.py --nerf_results_folder ./3models/models/TensoRF/log/tensorf_"$scene"_all/imgs_test_all \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_tensorrf --skip_count 16 --angle_threshold 24&

