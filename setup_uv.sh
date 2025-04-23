curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv NEURON822 --python=3.10.12
source NEURON822/bin/activate
git clone git@github.com:akaLIAN313/CMU_18879_project.git 
cd CMU_18879_project
uv pip install -r requirements.txt
nrnivmodl mechanisms #Compiling required Mechanisms for NEURON
