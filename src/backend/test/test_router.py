"""This is the test for backend API"""

def test_base(test_app):
    """Test response for base endpoint"""

    response = test_app.get('/cake_classifier')
    assert response.status_code == 200
    assert response.json() == {'Model Name': "resnet 50 + mobilenet"}

def test_env(test_app):
    '''Test response for env endpoint'''
    response = test_app.get('/cake_classifier/env')
    assert response.status_code == 200
    assert response.json() == {'Env': 'testing'}