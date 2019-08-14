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
import re
from core.model import ModelWrapper
from flask_restplus import fields
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from maxfw.core import MAX_API, PredictAPI


# set up parser for audio input data
input_parser = MAX_API.parser()
input_parser.add_argument('audio', type=FileStorage, location='files', required=True,
                          help="signed 16-bit PCM WAV audio file")
input_parser.add_argument('start_time', type=float, default=0,
                          help='The number of seconds into the audio file the prediction should start at.')
input_parser.add_argument('filter', required=False, action='split', help='List of labels to filter (optional)')

label_prediction = MAX_API.model('LabelPrediction', {
    'label_id': fields.String(required=False, description='Label identifier'),
    'label': fields.String(required=True, description='Audio class label'),
    'probability': fields.Float(required=True)
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(label_prediction), description='Predicted audio classes and probabilities')
})


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Predict audio classes from input data"""
        result = {'status': 'error'}

        args = input_parser.parse_args()

        if not re.match("audio/.*wav", str(args['audio'].mimetype)):
            e = BadRequest()
            e.data = {'status': 'error', 'message': 'Invalid file type/extension: ' + str(args['audio'].mimetype)}
            raise e

        audio_data = args['audio'].read()

        # Getting the predictions
        try:
            preds = self.model_wrapper._predict(audio_data, args['start_time'])
        except ValueError:
            e = BadRequest()
            e.data = {'status': 'error', 'message': 'Invalid start time: value outside audio clip'}
            raise e

        # Aligning the predictions to the required API format
        label_preds = [{'label_id': p[0], 'label': p[1], 'probability': p[2]} for p in preds]

        # Filter list
        if args['filter'] is not None and any(x.strip() != '' for x in args['filter']):
            label_preds = [x for x in label_preds if x['label'] in args['filter']]

        result['predictions'] = label_preds
        result['status'] = 'ok'

        return result
