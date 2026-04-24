# marl_sc

# $env:PYTHONPATH="C:\Users\Jakob Ehrenhuber\Desktop\marl-sc"     


ls -lah /home/jakobeh/projects/marl-sc/experiment_outputs/ | grep nfs 

rm -f /home/jakobeh/projects/marl-sc/experiment_outputs/BASELINES_single_3WH_2SKUS_SIMPLIFIED_SYMMETRIC/visualizations/.nfs*


rm -f /home/jakobeh/projects/marl-sc/experiment_outputs/WorkingConfig_Phase1.4/IPPO_Single_1WH_1SKUS_Agent_PSFalse_LogStdFloor-0.7_NN64/visualizations/visualization_final/.nfs*


# After the array job finishes:
sacct -j <JobID> --format=JobID,State,ExitCode | grep FAILED
# e.g. shows tasks 3, 11, 17 failed
sbatch --array=3,11,17 scripts/run_experiment_batch.sh --name MyFolder