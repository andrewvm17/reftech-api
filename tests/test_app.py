# tests/test_app.py

import os
import io
import pytest
from PIL import Image
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_get_lines_with_offside_image(client):
    """
    Test the /get-lines endpoint using an actual file (offside.png).
    Assumes 'offside.png' is located in the same directory as this test file
    or a known relative path.
    """
    # 1. Build the full path to 'offside.png' based on the current file's directory
    test_dir = os.path.dirname(__file__)
    offside_path = os.path.join(test_dir, 'offside.png')

    # 2. Make sure the file exists
    #    (Optional: you could assert or check existence; 
    #    if it doesn't exist, the open will fail anyway.)
    assert os.path.isfile(offside_path), f"offside.png not found at {offside_path}"

    # 3. Open the file in binary mode so we can send it as form data
    with open(offside_path, 'rb') as f:
        data = {
            'image': (f, 'offside.png')
        }

        # 4. Post to our endpoint
        response = client.post('/get-lines', content_type='multipart/form-data', data=data)

    # 5. Assert the response is 200 OK
    assert response.status_code == 200

    # 6. Parse JSON and assert keys
    json_data = response.get_json()
    assert 'x1' in json_data
    assert 'y1' in json_data
    assert 'x2' in json_data
    assert 'y2' in json_data
    assert 'slope' in json_data
