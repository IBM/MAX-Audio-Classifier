import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'Model Asset Exchange Microservice'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'audio_embeddings-tf-imagenet'
    assert metadata['name'] == 'audio_embeddings TensorFlow Model'
    assert metadata['description'] == 'audio_embeddings TensorFlow model trained on Audio Set'
    assert metadata['license'] == 'Apache 2.0'


def test_predict():
    model_endpoint = 'http://localhost:5000/model/predict'
    file_path = 'assets/birds1.wav'

    with open(file_path, 'rb') as file:
        file_form = {'audio': (file_path, file, 'audio/wav')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'

    assert response['predictions'][0]['label_id'] == '/m/015p6'
    assert response['predictions'][0]['label'] == 'Bird'
    assert response['predictions'][0]['probability'] > 0.4


if __name__ == '__main__':
    pytest.main([__file__])
