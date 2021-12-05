#rsync -avh --include="*.png" --exclude='*' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-RIBP/00_motivation/results/ 00_motivation/results/
#rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-RIBP/01_prior/results/ 01_prior/results/
rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-RIBP/02_linear_gaussian/results/ 02_linear_gaussian/results/
#rsync -avh --include="*.csv" --exclude='*' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/03_omniglot/ 03_omniglot/
#rsync -avh --include="*.png" --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RIBP/03_omniglot/ 03_omniglot/
#rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-RIBP/02_griffiths_ghahramani/results/ 02_griffiths_ghahramani/results/
