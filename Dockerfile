FROM codait/max-base

RUN wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audioset/vggish_model.ckpt && mv vggish_model.ckpt /workspace/assets/
RUN wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audioset/vggish_pca_params.npz && mv vggish_pca_params.npz /workspace/assets/
RUN wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audioset/classifier_model.h5 && mv classifier_model.h5 /workspace/assets/

# Python package versions
ARG numpy_version=1.13.1
ARG tf_version=1.8.0
ARG scipy_version=0.19.1
ARG six_version=1.10.0

RUN pip install numpy==${numpy_version} && \
    pip install tensorflow==${tf_version} && \
    pip install scipy==${scipy_version} && \
    pip install resampy && \
    pip install six==${six_version} && \
    pip install pandas && \
    pip install keras && \
    pip install h5py && \
    pip install json_tricks

COPY . /workspace

EXPOSE 5000

CMD python app.py
