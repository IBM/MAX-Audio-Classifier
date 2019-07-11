#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import pandas as pd
import tensorflow as tf
from . import vggish_input
from . import vggish_params
from . import vggish_postprocess
from . import vggish_slim
from config import MODEL_META_DATA as model_meta
from maxfw.model import MAXModelWrapper
from config import DEFAULT_EMBEDDING_CHECKPOINT, DEFAULT_PCA_PARAMS, DEFAULT_CLASSIFIER_MODEL


class ModelWrapper(MAXModelWrapper):
    """
    Contains core functions to generate embeddings and classify them.
    Also contains any helper function required.
    """

    MODEL_META_DATA = model_meta

    def __init__(self, embedding_checkpoint=DEFAULT_EMBEDDING_CHECKPOINT, pca_params=DEFAULT_PCA_PARAMS,
                 classifier_model=DEFAULT_CLASSIFIER_MODEL):
        # Initialize the classifier model
        self.session_classify = tf.keras.backend.get_session()
        self.classify_model = tf.keras.models.load_model(classifier_model, compile=False)

        # Initialize the vgg-ish embedding model
        self.graph_embedding = tf.Graph()
        with self.graph_embedding.as_default():
            self.session_embedding = tf.Session()
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.session_embedding, embedding_checkpoint)
            self.features_tensor = self.session_embedding.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.session_embedding.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        # Prepare a postprocessor to munge the vgg-ish model embeddings.
        self.pproc = vggish_postprocess.Postprocessor(pca_params)

        # Metadata
        self.indices = pd.read_csv('samples/class_labels_indices.csv')

    def generate_embeddings(self, wav_file):
        """
        Generates embeddings as per the Audioset VGG-ish model.
        Post processes embeddings with PCA Quantization
        Input args:
            wav_file   = /path/to/audio/in/wav/format.wav
        Returns:
                None.
        """
        examples_batch = vggish_input.wavfile_to_examples(wav_file)
        [embedding_batch] = self.session_embedding.run([self.embedding_tensor],
                                                       feed_dict={self.features_tensor: examples_batch})
        return self.pproc.postprocess(embedding_batch)

    def classify_embeddings(self, processed_embeddings):
        """
        Performs classification on PCA Quantized Embeddings.
        Input args:
            processed_embeddings = numpy array of shape (1,10,128), dtype=float32
        Returns:
            class_scores = Output probabilities for the 527 classes - numpy array of shape (1,527).
        """
        output_tensor = self.classify_model.output
        input_tensor = self.classify_model.input
        class_scores = output_tensor.eval(feed_dict={input_tensor: processed_embeddings}, session=self.session_classify)
        return class_scores

    def _predict(self, wav_file, time_stamp):
        """
        Driver function that performs all core tasks.
        Input args:
            wav_file = /path/to/audio_file.wav
        Returns:
            preds = numpy array of shape (527,) containing class probabilities.
        """
        # Step 1: Generate the embeddings.
        raw_embeddings = self.generate_embeddings(wav_file)
        # Step 2: Process the embeddings before sending it to the classifier.
        embeddings_processed = self.classifier_pre_process(raw_embeddings, time_stamp)
        # Step 3: Classify the embeddings.
        raw_preds = self.classify_embeddings(embeddings_processed)
        # Step 4: Post process the raw prediction vectors to a more interpretable format.
        preds = self.classifier_post_process(raw_preds[0])
        return preds

    def classifier_pre_process(self, embeddings, time_stamp):
        """
        Helper function to make sure input to classifier the model is of standard size.
        * Clips/Crops audio clips embeddings to start at time_stamp if not default and throws error if invalid
        * Augments audio embeddings shorter than 10 seconds (10x128 tensor) to a multiple of itself
        closest to 10 seconds.
        * Clips/Crops audio clips embeddings than 10 seconds to 10 seconds.
        * Converts dtype of embeddings from uint8 to float32

        Input args :
            embeddings = numpy array of shape (x,128) where x is any arbitrary whole number >1.
        Returns:
            embeddings = numpy array of shape (1,10,128), dtype=float32.
        """
        embeddings_ts = int(time_stamp / vggish_params.EXAMPLE_HOP_SECONDS)
        embeddings_len = embeddings.shape[0]
        if 0 < embeddings_ts < embeddings_len:
            end_ts = embeddings_ts + 10
            end_ts = end_ts if end_ts < embeddings_len else embeddings_len
            embeddings = embeddings[embeddings_ts:end_ts, :]
        elif embeddings_ts < 0 or embeddings_ts >= embeddings_len:
            raise ValueError

        embeddings_len = embeddings.shape[0]
        if embeddings_len < 10:
            while embeddings_len < 10:
                embeddings = np.stack((embeddings, embeddings))
                embeddings_len = embeddings.size / 128
            embeddings = embeddings.reshape((int(embeddings_len), 128))
        else:
            pass
        embeddings = embeddings[0:10, :].reshape([1, 10, 128])
        embeddings = self.uint8_to_float32(embeddings)
        return embeddings

    def classifier_post_process(self, raw_preds):
        """
        This function converts raw result vectors into a more interpretable format.
        Input args:
            raw_preds : numpy array of size (527,1) containing class scores.
        Returns :
            preds : list of (label_id,label,probability) tuples for top 5 class scores.
        """
        top_preds = raw_preds.argsort()[-5:][::-1]
        preds = [(self.indices.loc[top_preds[i]]['mid'], self.indices.loc[top_preds[i]]['display_name'],
                  raw_preds[top_preds[i]]) for i in range(len(top_preds))]
        return preds

    def uint8_to_float32(self, x):
        return (np.float32(x) - 128.) / 128.
