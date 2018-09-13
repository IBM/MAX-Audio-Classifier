from flask_restplus import Namespace, Resource, fields
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from config import MODEL_META_DATA
from core.backend import ModelWrapper
import os

api = Namespace('model', description='Model information and inference operations')

model_meta = api.model('ModelMetadata', {
    'id': fields.String(required=True, description='Model identifier'),
    'name': fields.String(required=True, description='Model name'),
    'description': fields.String(required=True, description='Model description'),
    'license': fields.String(required=False, description='Model license')
})


@api.route('/metadata')
class Model(Resource):
    @api.doc('get_metadata')
    @api.marshal_with(model_meta)
    def get(self):
        """Return the metadata associated with the model"""
        return MODEL_META_DATA


label_prediction = api.model('LabelPrediction', {
    'label_id': fields.String(required=False, description='Label identifier'),
    'label': fields.String(required=True, description='Audio class label'),
    'probability': fields.Float(required=True)
})

predict_response = api.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(label_prediction), description='Predicted audio classes and probabilities')
})

# set up parser for audio input data
audio_parser = api.parser()
audio_parser.add_argument('audio', type=FileStorage, location='files', required=True,
                          help="signed 16-bit PCM WAV audio file")


@api.route('/predict')
class Predict(Resource):
    mw = ModelWrapper()

    @api.doc('predict')
    @api.expect(audio_parser)
    @api.marshal_with(predict_response)
    def post(self):
        """Predict audio classes from input data"""
        result = {'status': 'error'}

        args = audio_parser.parse_args()
        audio_data = args['audio'].read()

        # clean up from earlier runs
        if os.path.exists("/audio.wav"):
            os.remove("/audio.wav")
        
        if '.wav' in str(args['audio']):
            file = open("/audio.wav", "wb")
            file.write(audio_data)
            file.close()
        else:
            e = BadRequest()
            e.data = {'status': 'error', 'message': 'Invalid file type/extension'}
            raise e

        # Getting the predictions
        preds = self.mw.predict("/audio.wav")
        
        # Aligning the predictions to the required API format
        label_preds = [{'label_id': p[0], 'label': p[1], 'probability': p[2]} for p in preds]
        result['predictions'] = label_preds
        result['status'] = 'ok'
        
        os.remove("/audio.wav")

        return result
