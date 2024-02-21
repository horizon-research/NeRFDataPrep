export scene=ignatius
export dsf=2
# gardne with cicero_dgo 6
python3 temporal_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_long/dvgo_"$scene"_0_2_long/render_test_fine_last \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path temp_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name temp_dgo --skip_count 6 &
# gardne with cicero_dgo 16
python3 temporal_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_long/dvgo_"$scene"_0_2_long/render_test_fine_last \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path temp_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name temp_dgo --skip_count 16 &

# gardne with cicero_dgo 30
python3 temporal_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_long/dvgo_"$scene"_0_2_long/render_test_fine_last \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path temp_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name temp_dgo --skip_count 30 &


# gardne with cicero_dgo 60
python3 temporal_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_long/dvgo_"$scene"_0_2_long/render_test_fine_last \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path temp_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name temp_dgo --skip_count 60 &

# gardne with cicero_dgo 180
python3 temporal_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_long/dvgo_"$scene"_0_2_long/render_test_fine_last \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path temp_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name temp_dgo --skip_count 180 &

# gardne with cicero_dgo 480
python3 temporal_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_long/dvgo_"$scene"_0_2_long/render_test_fine_last \
    --gt_folder "$scene"_900_10_seq/images_"$dsf" --depth_and_mask_folder "$scene"_900_10_seq/depths_masks_"$dsf" \
    --result_path temp_logs_long/ --item_name "$scene"_900_10_seq --meta_data_path "$scene"_900_10_seq/fix_norm_meta_val.pkl \
    --downscale_factor "$dsf" --method_name temp_dgo --skip_count 480 &
