import json
from unittest.mock import Mock

from PIL import Image
import pytest

from backend import workflow_integrations as integration


KEY = "test-secret-not-a-real-key-123456789"


def setup(tmp_path, monkeypatch, response):
    keyfile = tmp_path / "key.txt"
    keyfile.write_text(KEY, encoding="utf-8")
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.get.return_value = response
    session.post.return_value = response
    monkeypatch.setattr(integration, "_session", lambda:session)
    return keyfile, session


def test_credentials_stay_server_side_and_only_fixed_host(tmp_path, monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {"data":[{"id":"vision-model"},{"id":KEY}]}
    path, session = setup(tmp_path,monkeypatch,response)
    public = integration.configure(tmp_path,path)
    saved = integration.config_path(tmp_path).read_text()
    assert KEY not in saved and KEY not in json.dumps(public)
    assert str(path) not in json.dumps(public)
    assert public["munolith"]["models"] == ["vision-model"]
    assert session.get.call_args.args[0] == integration.BASE_URL+"/models"
    assert session.get.call_args.kwargs["allow_redirects"] is False


def test_failed_auth_never_returns_response_or_request_secret(tmp_path,monkeypatch):
    response=Mock(status_code=401,text=KEY)
    path,session=setup(tmp_path,monkeypatch,response)
    result=integration.configure(tmp_path,path)
    assert result["munolith"]["available"] is False
    assert "401" in result["munolith"]["error"]
    assert KEY not in json.dumps(result)
    session.get.side_effect=RuntimeError(KEY)
    assert KEY not in json.dumps(integration.configure(tmp_path,path))


def client(tmp_path,monkeypatch,response):
    path,session=setup(tmp_path,monkeypatch,response)
    integration.config_path(tmp_path).write_text(json.dumps({"available":True,"key_file_path":str(path),"models":["vision-model"]}))
    return integration.VisionClient(tmp_path,"vision-model"),session


def test_vlm_redacts_credential_echo_and_bounds_output(tmp_path,monkeypatch):
    response=Mock(status_code=200)
    response.json.return_value={"choices":[{"message":{"content":json.dumps({"actions":[{"label":"standing","score":.8,"explanation":KEY}]})}}]}
    model,session=client(tmp_path,monkeypatch,response)
    rows=model.predict(Image.new("RGB",(10,10)),"Observe actions")
    assert rows[0]["scope"]=="scene"
    assert KEY not in json.dumps(rows)
    assert session.post.call_args.kwargs["allow_redirects"] is False
    assert session.post.call_args.args[0]==integration.BASE_URL+"/chat/completions"


@pytest.mark.parametrize("score",[float("nan"),2,-1,"high"])
def test_invalid_scores_not_actions(tmp_path,monkeypatch,score):
    response=Mock(status_code=200)
    response.json.return_value={"choices":[{"message":{"content":json.dumps({"actions":[{"label":"standing","score":score}]})}}]}
    model,_=client(tmp_path,monkeypatch,response)
    with pytest.raises(ValueError,match="JSON"):
        model.predict(Image.new("RGB",(10,10)),"Observe")


def test_vlm_network_exception_cannot_leak_headers(tmp_path,monkeypatch):
    model,session=client(tmp_path,monkeypatch,Mock())
    session.post.side_effect=RuntimeError(KEY)
    with pytest.raises(ValueError) as error:
        model.predict(Image.new("RGB",(10,10)),"Observe")
    assert KEY not in str(error.value)
