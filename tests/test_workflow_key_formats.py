"""Credential parsing regressions; all values are synthetic and remain local."""

import pytest

from backend import workflow_integrations as integration


KEY = "synthetic-test-credential-0123456789"
SECOND_KEY = "another-synthetic-credential-9876543210"


def key_file(tmp_path, content):
    path = tmp_path / "synthetic-credentials.txt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(KEY, id="raw-key"),
        pytest.param(f'"{KEY}"', id="quoted-raw-key"),
        pytest.param(f"API_KEY={KEY}", id="plain-env"),
        pytest.param(f"LMSTUDIO_GATEWAY_API_KEY={KEY}", id="gateway-env"),
        pytest.param(f"OPENAI_API_KEY={KEY}", id="provider-env"),
        pytest.param(f"SERVICE_2_ACCESS_TOKEN={KEY}", id="namespaced-access-token"),
        pytest.param(f"ACCESS_TOKEN={KEY}", id="access-token"),
        pytest.param(f"TOKEN={KEY}", id="legacy-token"),
        pytest.param(f"Authorization: Bearer {KEY}", id="authorization-header"),
        pytest.param(f"Bearer {KEY}", id="bearer-value"),
        pytest.param(f"export LMSTUDIO_GATEWAY_API_KEY={KEY}", id="export-env"),
        pytest.param(f"\ufeff  export LMSTUDIO_GATEWAY_API_KEY = '{KEY}'  \n", id="bom-export-quotes"),
        pytest.param(f'LMSTUDIO_GATEWAY_API_KEY="{KEY}"', id="double-quoted-env"),
        pytest.param(f"# Synthetic fixture only\n\nLMSTUDIO_GATEWAY_API_KEY={KEY}\n", id="comments-and-blank-lines"),
    ],
)
def test_read_key_accepts_supported_credential_formats(tmp_path, content):
    assert integration.read_key(key_file(tmp_path, content)) == KEY


@pytest.mark.parametrize(
    "raw_key",
    [
        pytest.param("c3ludGhldGljLWJhc2U2NC10ZXN0LWtleQ==", id="double-padding"),
        pytest.param("c3ludGhldGljLXRlc3Qta2V5LXN0cmluZw=", id="single-padding"),
        pytest.param("synthetic+credential/with_padding==", id="base64-alphabet"),
        pytest.param("ABCDEFGHIJKLMNOPQRSTUVU=", id="uppercase-name-single-padding"),
        pytest.param("ABCDEFGHIJKLMNOPQRSTUQ==", id="uppercase-name-double-padding"),
    ],
)
@pytest.mark.parametrize("prefix", ["", "LMSTUDIO_GATEWAY_API_KEY="])
def test_read_key_preserves_base64_padding(tmp_path, raw_key, prefix):
    assert integration.read_key(key_file(tmp_path, prefix + raw_key)) == raw_key


def test_read_key_ignores_noncredential_env_metadata(tmp_path):
    content = (
        "# This file uses dotenv syntax.\n"
        "API_BASE_URL=https://example.invalid/v1\n"
        f"LMSTUDIO_GATEWAY_API_KEY={KEY}\n"
    )
    assert integration.read_key(key_file(tmp_path, content)) == KEY


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(f"{KEY}\n{SECOND_KEY}\n", id="two-raw-keys"),
        pytest.param(f"LMSTUDIO_GATEWAY_API_KEY={KEY}\nOTHER_ACCESS_TOKEN={SECOND_KEY}\n", id="two-env-keys"),
        pytest.param(f"API_KEY={KEY}\nTOKEN={KEY}\n", id="duplicate-credentials"),
        pytest.param(f"API_KEY={KEY}\n{SECOND_KEY}\n", id="mixed-credentials"),
    ],
)
def test_read_key_rejects_multiple_credentials_without_echoing_them(tmp_path, content, capsys):
    with pytest.raises(ValueError) as error:
        integration.read_key(key_file(tmp_path, content))
    captured = capsys.readouterr()
    public_output = str(error.value) + captured.out + captured.err
    assert KEY not in public_output
    assert SECOND_KEY not in public_output


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("", id="empty-file"),
        pytest.param("\ufeff# Comment only\n", id="comment-only"),
        pytest.param("LMSTUDIO_GATEWAY_API_KEY=", id="empty-env-value"),
        pytest.param('LMSTUDIO_GATEWAY_API_KEY=""', id="empty-quoted-value"),
        pytest.param("LMSTUDIO_GATEWAY_API_KEY=too-short", id="short-value"),
        pytest.param(f"LMSTUDIO_GATEWAY_API_KEY={KEY} invalid whitespace", id="invalid-value"),
    ],
)
def test_read_key_rejects_missing_or_invalid_value_without_echoing_contents(tmp_path, content, capsys):
    with pytest.raises(ValueError) as error:
        integration.read_key(key_file(tmp_path, content))
    captured = capsys.readouterr()
    public_output = str(error.value) + captured.out + captured.err
    assert KEY not in public_output
    assert "too-short" not in public_output
    assert "invalid whitespace" not in public_output
