import pytest
import requests


def test_response():
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
