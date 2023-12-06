FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install git -y 

RUN /opt/conda/bin/conda install jupyter jupyter_server>=1.11.0 scipy -y \
  && /opt/conda/bin/pip install fire tqdm scikit-learn pandas matplotlib  \
  && /opt/conda/bin/pip install pyDOE sobol_seq \
  && /opt/conda/bin/pip install torchnet \
  && /opt/conda/bin/pip install torchdiffeq \
  && /opt/conda/bin/pip install sympy 
