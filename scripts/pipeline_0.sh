#!/bin/bash
set -e 

ARG1="2024-10-14_01_ESPM113"

ARG2="/home/pmateosaparicio/data/Repository/ESPM113/2024-10-14_02_ESPM113" #stimuli session

ARG3="/home/pmateosaparicio/data/Repository/ESPM113/2024-10-14_01_ESPM113" #sleep session


run_step() {
   local step_name="$1"
   local command="$2"
   local cores=8  # number of cores
   
   echo "$step_name" | tee -a log_${ARG1}.txt
   start_time=$(date +%s)
   
   taskset -c 0-$((cores-1)) $command
   
   end_time=$(date +%s)
   duration=$((end_time - start_time))
   echo "$step_name completed in ${duration}s using ${cores} cores" | tee -a log_${ARG1}.txt
}

# Pipeline
> log_${ARG1}.txt

# run_step "0 prepare" "python 0_prepare.py --mouse_id $ARG1"
# run_step "1 1 data processing" "python 1_1_data_processing.py --mouse_id $ARG1 --stimuli_session $ARG2 --sleep_session $ARG3"
# run_step "1 2 eda" "python 1_2_eda.py --mouse_id $ARG1"
# run_step "2 1 raster map calculation" "python 2_1_raster_map_calculation.py --mouse_id $ARG1"
# run_step "2 2 raster maps plotting" "python 2_2_raster_maps_plotting.py --mouse_id $ARG1"
# # run_step "2 3 raster maps enhanced" "python 2_3_raster_maps_enhanced.py --mouse_id $ARG1"
# run_step "2 4 raster maps matrix" "python 2_4_raster_maps_matrix.py --mouse_id $ARG1"
# run_step "3 1 corr matrix calculation" "python 3_1_corr_matrix_calculation.py --mouse_id $ARG1"
# run_step "3 2 corr matrix sampled calculation" "python 3_2_corr_matrix_sampled_calculation.py --mouse_id $ARG1"
run_step "3 3 corr matrices plotting" "python 3_3_corr_matrices_plotting.py --mouse_id $ARG1"
# run_step "3 4 big corr matrix plotting" "python 3_4_big_corr_matrix_plotting.py --mouse_id $ARG1"
# run_step "4 1 pca calculation" "python 4_1_pca_calculation.py --mouse_id $ARG1"
# run_step "4 2 pca projection" "python 4_2_pca_projection.py --mouse_id $ARG1"
# # run_step "4 3 pca running speed" "python 4_3_pca_running_speed.py --mouse_id $ARG1"
# run_step "5 1 pca sampled calculation" "python 5_1_pca_sampled_calculation.py --mouse_id $ARG1"
# run_step "5 2 pca sampled plot" "python 5_2_pca_sampled_plot.py --mouse_id $ARG1"
# run_step "5 3 pca number of pcs x time" "python 5_3_pca_number_of_pcs_x_time.py --mouse_id $ARG1"
# run_step "6 1 umap calculation" "python 6_1_umap_calculation.py --mouse_id $ARG1"
# run_step "6 2 umap plotting" "python 6_2_umap_plotting.py --mouse_id $ARG1"
# run_step "7 1 Classifier training" "python 7_1_classifier_training.py --mouse_id $ARG1"

echo "Pipeline finished!" | tee -a log_${ARG1}.txt