export scene=garden
export dsf=4
# gardne with cicero_instant_ngp 6
python3 warping_evaluation_thresh.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 6 --angle_threshold 4&

# gardne with cicero_instant_ngp 6
python3 warping_evaluation_thresh.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 6 --angle_threshold 8&

# gardne with cicero_instant_ngp 6
python3 warping_evaluation_thresh.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 6 --angle_threshold 12&

# gardne with cicero_instant_ngp 6
python3 warping_evaluation_thresh.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 6 --angle_threshold 16&

# gardne with cicero_instant_ngp 6
python3 warping_evaluation_thresh.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 6 --angle_threshold 20&

# gardne with cicero_instant_ngp 6
python3 warping_evaluation_thresh.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs_thresh/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 6 --angle_threshold 24&
