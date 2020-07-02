#@velascoluis use https://github.com/rocker-org/rocker
FROM rocker/rstudio:latest
RUN mkdir -p /app
COPY src/secret/* /app/
COPY src/* /app/
#@velascoluis
#Nedeed for xgboost - Error: the #include file "zlib.h" is not found.
RUN sudo apt-get install libz-dev
RUN Rscript /app/install_packages.R