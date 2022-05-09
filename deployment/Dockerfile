FROM pytorch/torchserve

USER root

RUN pip3 install sentence-transformers torchserve
RUN pip3 install torch-model-archiver -q

USER model-server

EXPOSE 8080
EXPOSE 8081
EXPOSE 8082
EXPOSE 80

COPY . .
COPY config.properties config.properties
RUN mkdir -p model_store

CMD ["torchserve", "--start", "--model-store", "model_store"]
#CMD ["torchserve", "--start", "--model-store", "model_store", "--models", "bert=bert.mar"]
#CMD ["torchserve", "--start", "--model-store", "model_store", "--models", "nn=nn.mar bert=bert.mar"]
