#!/bin/bash
# 
# Send all my jobs in a swarm

for n in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900
do
	jobfile="runjob.sh"

	echo "submitting with N=$n"

	echo "#!/bin/sh" > $jobfile
	echo "#SBATCH --account=theory" >> $jobfile
	echo "#SBATCH --job-name=geom" >> $jobfile
	echo "#SBATCH -c 1" >> $jobfile
	echo "#SBATCH --time=1-11:59:00" >> $jobfile
	echo "#SBATCH --mem-per-cpu=1gb" >> $jobfile

	echo "module load anaconda/3-5.1" >> $jobfile
	echo "source activate pete" >> $jobfile
	
	echo "python bert/pairwise_swapsimilarity.py -l $n" >> $jobfile

	echo "date" >> $jobfile

	sbatch $jobfile
	echo "waiting"
	sleep 1s
done

