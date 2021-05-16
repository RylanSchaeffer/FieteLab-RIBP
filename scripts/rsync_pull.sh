echo -n password:
read -s password


sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/exp_00_ibp_prior/plots/ exp_00_ibp_prior/plots/
sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/exp_00_ibp_prior/ exp_00_ibp_prior/