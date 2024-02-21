export scene=ignatius
export dsf=2
# ignatius with cicero_instant_ngp 6
python3 warping_evaluation.py --nerf_results_folder "$scene"_900_10_seq/ingp_256_35000_base_snapshots/ \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 6 &

# ignatius with cicero_instant_ngp 16
python3 warping_evaluation.py --nerf_results_folder "$scene"_900_10_seq/ingp_256_35000_base_snapshots/ \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 16 &

# ignatius with cicero_instant_ngp 30
python3 warping_evaluation.py --nerf_results_folder "$scene"_900_10_seq/ingp_256_35000_base_snapshots/ \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 30 &

# ignatius with cicero_instant_ngp 60
python3 warping_evaluation.py --nerf_results_folder "$scene"_900_10_seq/ingp_256_35000_base_snapshots/ \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 60 &

# ignatius with cicero_instant_ngp 180
python3 warping_evaluation.py --nerf_results_folder "$scene"_900_10_seq/ingp_256_35000_base_snapshots/ \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 180 &

# ignatius with cicero_instant_ngp 480
python3 warping_evaluation.py --nerf_results_folder "$scene"_900_10_seq/ingp_256_35000_base_snapshots/ \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path cicero_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name cicero_instant_ngp --skip_count 480 &