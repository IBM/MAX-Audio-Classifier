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

# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Audio Classifier'
API_DESC = 'Identify sounds in short audio clips'
API_VERSION = '1.2.0'

# default model
MODEL_NAME = 'audio_embeddings'
MODEL_LICENSE = 'Apache 2.0'

MODEL_META_DATA = {
    'id': '{}-tf-imagenet'.format(MODEL_NAME.lower()),
    'name': '{} TensorFlow Model'.format(MODEL_NAME),
    'description': '{} TensorFlow model trained on Audio Set'.format(MODEL_NAME),
    'type': 'image_classification',
    'license': '{}'.format(MODEL_LICENSE)
}

DEFAULT_EMBEDDING_CHECKPOINT = "assets/vggish_model.ckpt"
DEFAULT_PCA_PARAMS = "assets/vggish_pca_params.npz"
DEFAULT_CLASSIFIER_MODEL = "assets/classifier_model.h5"
