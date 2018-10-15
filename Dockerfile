FROM codait/max-base

RUN wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audio-classifier/assets.tar.gz && mv assets.tar.gz /workspace/assets/
RUN tar -x -C /workspace/assets -f /workspace/assets/assets.tar.gz -v && rm /workspace/assets/assets.tar.gz

COPY requirements.txt /workspace
RUN pip install -r requirements.txt

COPY . /workspace
RUN md5sum -c md5sums.txt # check file integrity

EXPOSE 5000

CMD python app.py
