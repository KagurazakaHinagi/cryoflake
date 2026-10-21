"""Read-only local viewer routes, asset packaging, and HTTP isolation."""

from io import BytesIO
from types import SimpleNamespace
from wsgiref.util import setup_testing_defaults

import pytest

pytest.importorskip("bottle")

from cryoflake.visualization import server as server_module  # noqa: E402


@pytest.fixture
def bundle(tmp_path):
    artifact_data = {
        "structure_path": ("structure.cif", b"data_structure\n"),
        "density_path": ("density.mrc", b"\x00\x01\x02\xff"),
        "sequence_path": ("sequence.fasta", b">POLYMER_0\nACD\n"),
        "summary_path": ("summary.json", b'{"schema_version": 1}\n'),
    }
    paths = {}
    for name, (filename, contents) in artifact_data.items():
        paths[name] = tmp_path / filename
        paths[name].write_bytes(contents)
    (tmp_path / "private.json").write_text('{"private":true}')
    return SimpleNamespace(**paths)


def _request(application, path, *, method="GET", host="127.0.0.1:8888", origin=None):
    environ = {}
    setup_testing_defaults(environ)
    environ.update(
        {
            "PATH_INFO": path,
            "REQUEST_METHOD": method,
            "HTTP_HOST": host,
            "SERVER_PORT": "8888",
            "wsgi.input": BytesIO(),
        }
    )
    if origin:
        environ["HTTP_ORIGIN"] = origin
    result = {}

    def start_response(status, headers, exc_info=None):
        result["status"] = int(status.split()[0])
        result["headers"] = {name.lower(): value for name, value in headers}

    body = application(environ, start_response)
    try:
        result["body"] = b"".join(body)
    finally:
        if hasattr(body, "close"):
            body.close()
    return result


@pytest.mark.parametrize(
    "route,attribute,content_type",
    [
        ("/structure.cif", "structure_path", "chemical/x-mmcif"),
        ("/density.mrc", "density_path", "application/octet-stream"),
        ("/sequence.fasta", "sequence_path", "text/plain"),
        ("/summary.json", "summary_path", "application/json"),
    ],
)
def test_artifact_routes_serve_exact_reconstructed_data(
    bundle, route, attribute, content_type
):
    application = server_module._visualization_application(bundle)
    result = _request(application, route)
    assert result["status"] == 200
    assert result["body"] == getattr(bundle, attribute).read_bytes()
    assert result["headers"]["content-type"].startswith(content_type)
    assert result["headers"]["content-length"] == str(len(result["body"]))
    assert "access-control-allow-origin" not in result["headers"]


@pytest.mark.parametrize(
    "route,content_type,expected",
    [
        ("/", "text/html", b"<title>cryoFlake"),
        ("/app.css", "text/css", b".viewer-container"),
        ("/app.js", "text/javascript", b"loadVolumeFromUrl"),
    ],
)
def test_packaged_viewer_assets_are_served(bundle, route, content_type, expected):
    application = server_module._visualization_application(bundle)
    result = _request(application, route)
    assert result["status"] == 200
    assert result["headers"]["content-type"].startswith(content_type)
    assert expected in result["body"]


@pytest.mark.parametrize(
    "route",
    [
        "/private.json",
        "/metadata.json",
        "/../private.json",
        "/%2e%2e/private.json",
        "/static/../../private.json",
        "/density.mrc/../private.json",
        "/static/",
    ],
)
def test_only_allowlisted_routes_are_available(bundle, route):
    result = _request(server_module._visualization_application(bundle), route)
    assert result["status"] == 404
    assert b'{"private":true}' not in result["body"]


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def test_viewer_rejects_mutating_and_cross_origin_methods(bundle, method):
    result = _request(
        server_module._visualization_application(bundle), "/summary.json", method=method
    )
    assert result["status"] == 405
    assert bundle.summary_path.read_bytes() == b'{"schema_version": 1}\n'


@pytest.mark.parametrize("route", ["/", "/app.css", "/density.mrc"])
def test_head_returns_headers_without_body(bundle, route):
    result = _request(
        server_module._visualization_application(bundle), route, method="HEAD"
    )
    assert result["status"] == 200
    assert result["body"] == b""
    assert int(result["headers"]["content-length"]) > 0


@pytest.mark.parametrize(
    "host,origin,expected_status",
    [
        ("127.0.0.1:8888", None, 200),
        ("localhost:8888", "http://localhost:8888", 200),
        ("127.0.0.1:8888", "http://127.0.0.1:8888", 200),
        ("rebound.example:8888", None, 403),
        ("127.0.0.1:9999", None, 403),
        ("localhost.attacker.example:8888", None, 403),
        ("127.0.0.1:8888", "http://attacker.example", 403),
        ("127.0.0.1:8888", "null", 403),
        ("127.0.0.1:8888", "http://localhost:9999", 403),
    ],
)
def test_foreign_hosts_and_origins_cannot_read_artifact(
    bundle, host, origin, expected_status
):
    result = _request(
        server_module._visualization_application(bundle),
        "/summary.json",
        host=host,
        origin=origin,
    )
    assert result["status"] == expected_status
    if expected_status == 403:
        assert bundle.summary_path.read_bytes() not in result["body"]


@pytest.mark.parametrize("route", ["/", "/summary.json", "/unknown"])
def test_http_headers_keep_artifact_data_local(bundle, route):
    headers = _request(server_module._visualization_application(bundle), route)[
        "headers"
    ]
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["x-frame-options"] == "DENY"
    assert headers["cross-origin-resource-policy"] == "same-origin"
    policy = headers["content-security-policy"]
    assert "connect-src 'self';" in policy
    # The pinned Mol* browser bundle dynamically compiles molecular queries.
    assert "script-src 'self' 'unsafe-eval' https://cdn.jsdelivr.net;" in policy
    assert "script-src 'self' 'unsafe-inline'" not in policy
    assert "frame-ancestors 'none'" in policy


def test_server_binds_only_loopback_and_supports_ephemeral_port(bundle, monkeypatch):
    captured = {}
    expected_server = object()

    def make_server(host, port, application, *, server_class, handler_class):
        captured.update(
            host=host,
            port=port,
            application=application,
            server_class=server_class,
            handler_class=handler_class,
        )
        return expected_server

    monkeypatch.setattr(server_module, "make_server", make_server)
    assert server_module.create_visualization_server(bundle) is expected_server
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 0
    assert captured["server_class"].daemon_threads is True
    assert _request(captured["application"], "/density.mrc")["status"] == 200


@pytest.mark.parametrize("port", [-1, 65536])
def test_server_rejects_invalid_port_before_binding(bundle, port):
    with pytest.raises(ValueError, match="port must be between"):
        server_module.create_visualization_server(bundle, port=port)
