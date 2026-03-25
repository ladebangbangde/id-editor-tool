from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_formal_wear_endpoint_returns_offline_placeholder() -> None:
    response = client.post('/formal-wear', data={'style': 'formal', 'color': 'black'})

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is False
    assert payload['message'] == '换装功能已下线'
    assert payload['error']['code'] == 'FORMAL_WEAR_OFFLINE'
    assert payload['data']['status'] == 'offline'
