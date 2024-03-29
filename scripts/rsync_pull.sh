rsync -avh --include="*.png" rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-RIBP/00_motivation/results/ 00_motivation/results/
#rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-RIBP/01_prior/results/ 01_prior/results/
#rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-RIBP/02_linear_gaussian/results/ 02_linear_gaussian/results/
#rsync -avh --include="*.csv" --exclude='*' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/03_omniglot/ 03_omniglot/
rsync -avh --include="*.png" --include="*.csv" --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/03_omniglot/results_data=100/ 03_omniglot/results_data=100/
rsync -avh --include="*.png" --include="*.csv" --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/04_cancer_gene_expression/results_data=100/plots/ 04_cancer_gene_expression/results_data=100/plots/
#rsync -avh --include="*.png" --include="*.csv" --exclude='*' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/04_cancer_gene_expression/results_data=1000/ 04_cancer_gene_expression/results_data=1000/
rsync -avh --include="*.png" --include="*.csv" --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/05_diabetes_hospitals/results_data=100/plots/ 05_diabetes_hospitals/results_data=100/plots/
