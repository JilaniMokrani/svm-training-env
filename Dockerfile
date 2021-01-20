FROM jilani95/data-preprocessing:latest
RUN pip3 install scikit-learn
ENV DATA_SOURCE=/data
ENV MODEL_PATH=/models
COPY ./main.py .
CMD python3 main.py && bash