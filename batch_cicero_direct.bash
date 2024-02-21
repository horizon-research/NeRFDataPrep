export scene=garden
export dsf=4
# gardne with cicero_dgo 6
python3 warping_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_all/dvgo_"$scene"_0_2_all/render_test_fine_last \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_dgo --skip_count 6 &
# gardne with cicero_dgo 16
python3 warping_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_all/dvgo_"$scene"_0_2_all/render_test_fine_last \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_dgo --skip_count 16 &



export scene=bonsai
export dsf=4
# gardne with cicero_dgo 6
python3 warping_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_all/dvgo_"$scene"_0_2_all/render_test_fine_last \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_dgo --skip_count 6 &
# gardne with cicero_dgo 16
python3 warping_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_all/dvgo_"$scene"_0_2_all/render_test_fine_last \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_dgo --skip_count 16 &


export scene=ignatius
export dsf=2
# gardne with cicero_dgo 6
python3 warping_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_all/dvgo_"$scene"_0_2_all/render_test_fine_last \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_dgo --skip_count 6 &
# gardne with cicero_dgo 16
python3 warping_evaluation.py --nerf_results_folder ./3models/models/DirectVoxGO/logs/"$scene"_0_2_all/dvgo_"$scene"_0_2_all/render_test_fine_last \
    --gt_folder "$scene"/images_"$dsf" --depth_and_mask_folder "$scene"/depths_masks_"$dsf" \
    --result_path cicero_logs/ --item_name "$scene" --meta_data_path "$scene"/fix_norm_meta.pkl \
    --downscale_factor "$dsf" --method_name cicero_dgo --skip_count 16 &