How-To use R in Python:

# Non-Admin install of R will need R home added to (user) PATH 
# R, user install, on Windows: '%USERPROFILE%\Documents\R\R-4.1.3' ## (YMMV)

# one-time
#
conda create -n orange38R python=3.8
conda activate orange38R
conda install ipython
conda install orange3
conda install jupyter notebook
cd '%USERPROFILE%\orange3-conformal-master' # https://github.com/biolab/orange3-conformal
python setup.py develop
##conda install rpy2 # 2.9.x is quite lacking compared to 3.5+ for pandas, etc.
# use source
cd '%USERPROFILE%\rpy2-master' # https://github.com/rpy2/rpy2
python setup.py install


# usage
#
ipython # IDE/Jupyter Notebook/etc.
import rpy2
import rpy2.robjects.packages as rpackages # importing rpy2's package module
print(rpy2.__version__) # 3.5.3
rpy2.robjects.r['pi'] # 3.141593
utils = rpackages.importr('utils') # importing R's utility package
utils.chooseCRANmirror(ind=1) # selecting first mirror in list for R packages
packnames = ('ggplot2', 'BCRA')
from rpy2.robjects.vectors import StrVector
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
	utils.install_packages(StrVector(names_to_install))
rpackages.isinstalled('BCRA') # True
bcra = importr('BCRA')
bcra.__version__ # 2.1.2

# Note: R uses '.' (dot) to separate words in symbols, 
# 	similar to '_' in Python. Because, in Python,
#	dot means "attribute in a namespace", importr
# 	may sometimes translate '.' to '_'