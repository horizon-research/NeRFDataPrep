export scene=garden
export dsf=4
# gardne with temp_instant_ngp 16
python3 temporal_evaluation.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path temp_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name temp_instant_ngp --skip_count 16 &


export scene=bonsai
export dsf=4
# gardne with temp_instant_ngp 16
python3 temporal_evaluation.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path temp_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name temp_instant_ngp --skip_count 16 &


export scene=ignatius
export dsf=2
# gardne with temp_instant_ngp 16
python3 temporal_evaluation.py --nerf_results_folder "$scene"/ingp_256_35000_base_snapshots_all/ \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path temp_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name temp_instant_ngp --skip_count 16 &