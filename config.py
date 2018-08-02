# Flask settings 
DEBUG=False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER=False

# Application settings

# API metadata
API_TITLE = 'Model Asset Exchange Server'
API_DESC = 'An API for serving models'
API_VERSION = '0.1'

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

DEFAULT_EMBEDDING_CHECKPOINT = "/workspace/assets/vggish_model.ckpt"
DEFAULT_PCA_PARAMS = "/workspace/assets/vggish_pca_params.npz"
DEFAULT_CLASSIFIER_MODEL = "/workspace/assets/classifier_model.h5"