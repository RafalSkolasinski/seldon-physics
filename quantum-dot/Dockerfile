FROM seldonio/seldon-core-s2i-python37:0.19-SNAPSHOT


# conda environment defined in conda.yml so s2i will not try to install it again
copy conda.yml /microservice
RUN conda env create --name microservice --file conda.yml
