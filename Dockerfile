FROM codait/max-base

RUN wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audioset/vggish_model.ckpt && mv vggish_model.ckpt /workspace/assets/
RUN wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audioset/vggish_pca_params.npz && mv vggish_pca_params.npz /workspace/assets/
RUN wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audioset/classifier_model.h5 && mv classifier_model.h5 /workspace/assets/

COPY requirements.txt /workspace
RUN pip install -r requirements.txt

COPY . /workspace

EXPOSE 5000

CMD python app.py
