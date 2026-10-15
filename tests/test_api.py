from pathlib import Path
import sys
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from api.app import app

client=TestClient(app)

def test_viewer_is_served():
    response=client.get('/')
    assert response.status_code == 200
    assert '3D Cardiovascular Risk Explorer' in response.text

def test_schema_has_no_label_inputs():
    response=client.get('/schema')
    assert response.status_code == 200
    names={feature['name'] for feature in response.json()['features']}
    assert not names.intersection({'Cath','LAD','LCX','RCA'})

def test_predict_contract_for_example():
    example=client.get('/examples').json()['examples'][0]
    response=client.post('/predict',json={'features':example['features'],'include_explanations':False})
    assert response.status_code == 200
    body=response.json()
    assert set(body['targets']) == {'CAD','LAD','LCX','RCA'}
    assert all(0 <= target['probability'] <= 1 for target in body['targets'].values())

def test_shap_explanation_for_one_target():
    from shap_explain import explain_features
    example=client.get('/examples').json()['examples'][0]
    explanation=explain_features('CAD',example['features'])
    assert explanation['method'] == 'LinearExplainer'
    assert len(explanation['contributions']) > 0
