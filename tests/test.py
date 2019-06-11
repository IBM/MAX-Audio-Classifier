import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Audio Classifier'
    assert json.get('info').get('version') == '1.1.0'
    assert json.get('info').get('description') == 'Identify sounds in short audio clips'


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


def test_empty_filter():

    model_endpoint = 'http://localhost:5000/model/predict?filter='
    file_path = 'assets/gunshots.wav'

    with open(file_path, 'rb') as file:
        file_form = {'audio': (file_path, file, 'audio/wav')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'
    assert len(response['predictions']) >= 5

    assert response['predictions'][0]['label_id'] == '/m/032s66'
    assert response['predictions'][0]['label'] == 'Gunshot, gunfire'
    assert response['predictions'][0]['probability'] > 0.5


def test_multi_empty_filter():

    model_endpoint = 'http://localhost:5000/model/predict?filter=,,'
    file_path = 'assets/gunshots.wav'

    with open(file_path, 'rb') as file:
        file_form = {'audio': (file_path, file, 'audio/wav')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'
    assert len(response['predictions']) >= 5

    assert response['predictions'][0]['label_id'] == '/m/032s66'
    assert response['predictions'][0]['label'] == 'Gunshot, gunfire'
    assert response['predictions'][0]['probability'] > 0.5


def test_filter():

    model_endpoint = 'http://localhost:5000/model/predict?filter=Cap%20gun'
    file_path = 'assets/gunshots.wav'

    with open(file_path, 'rb') as file:
        file_form = {'audio': (file_path, file, 'audio/wav')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'
    assert len(response['predictions']) == 1

    assert response['predictions'][0]['label_id'] == '/m/073cg4'
    assert response['predictions'][0]['label'] == 'Cap gun'
    assert response['predictions'][0]['probability'] > 0.2


def test_multi_filter():

    model_endpoint = 'http://localhost:5000/model/predict?filter=Clang,Ding'
    file_path = 'assets/gunshots.wav'

    with open(file_path, 'rb') as file:
        file_form = {'audio': (file_path, file, 'audio/wav')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'
    assert len(response['predictions']) == 2

    assert response['predictions'][0]['label_id'] == '/m/07rv4dm'
    assert response['predictions'][0]['label'] == 'Clang'
    assert response['predictions'][0]['probability'] > 0.1

    assert response['predictions'][1]['label_id'] == '/m/07phxs1'
    assert response['predictions'][1]['label'] == 'Ding'
    assert response['predictions'][1]['probability'] > 0.09


if __name__ == '__main__':
    pytest.main([__file__])
