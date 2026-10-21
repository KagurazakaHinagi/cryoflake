"""A small, read-only, loopback server for one reconstructed artifact."""

from __future__ import annotations

from importlib.resources import files
from socketserver import ThreadingMixIn
from typing import TYPE_CHECKING
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from bottle import Bottle, HTTPError, request, response, static_file

if TYPE_CHECKING:
    from cryoflake.visualization.reconstruction import VisualizationBundle


class VisualizationServer(ThreadingMixIn, WSGIServer):
    """Serve concurrent local downloads without keeping shutdown waiting."""

    daemon_threads = True


class _QuietRequestHandler(WSGIRequestHandler):
    def log_message(self, format, *args):
        """Keep the CLI's URL and shutdown instructions readable."""


def _visualization_application(bundle: VisualizationBundle) -> Bottle:
    application = Bottle()
    artifact_routes = {
        "/structure.cif": (bundle.structure_path, "chemical/x-mmcif"),
        "/density.mrc": (bundle.density_path, "application/octet-stream"),
        "/sequence.fasta": (bundle.sequence_path, "text/plain; charset=UTF-8"),
        "/summary.json": (bundle.summary_path, "application/json"),
    }
    static_routes = {
        "/": ("index.html", "text/html; charset=UTF-8"),
        "/app.css": ("app.css", "text/css; charset=UTF-8"),
        "/app.js": ("app.js", "text/javascript; charset=UTF-8"),
    }

    @application.hook("before_request")
    def restrict_requests():
        # A foreign Host could otherwise expose local data through DNS rebinding.
        host = request.environ.get("HTTP_HOST", "")
        port = request.environ.get("SERVER_PORT", "")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            raise HTTPError(403, "This visualization is available on loopback only.")
        origin = request.headers.get("Origin")
        if origin and origin not in {
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
        }:
            raise HTTPError(403, "Cross-origin requests are not allowed.")

    @application.hook("after_request")
    def security_headers():
        response.set_header("Cache-Control", "no-store")
        response.set_header("X-Content-Type-Options", "nosniff")
        response.set_header("X-Frame-Options", "DENY")
        response.set_header("Referrer-Policy", "no-referrer")
        response.set_header("Cross-Origin-Resource-Policy", "same-origin")
        response.set_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            # Mol* compiles model queries with Function() in its pinned bundle.
            # Artifact values never become inline scripts; downloads remain local.
            "script-src 'self' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "worker-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )

    def serve_artifact():
        artifact_path, content_type = artifact_routes[request.path]
        return static_file(
            artifact_path.name,
            root=str(artifact_path.parent),
            mimetype=content_type,
        )

    def serve_static():
        filename, content_type = static_routes[request.path]
        response.content_type = content_type
        return (
            files("cryoflake.visualization").joinpath("static", filename).read_bytes()
        )

    for route in artifact_routes:
        application.route(route, method=["GET", "HEAD"], callback=serve_artifact)
    for route in static_routes:
        application.route(route, method=["GET", "HEAD"], callback=serve_static)
    return application


def create_visualization_server(
    bundle: VisualizationBundle, *, port: int = 0
) -> VisualizationServer:
    """Bind a read-only viewer to 127.0.0.1; the caller controls its lifetime.

    Port zero selects an available port. Only the viewer assets and the four
    reconstructed downloads are served; the artifact directory is never exposed.
    """
    if not 0 <= port <= 65535:
        raise ValueError("Visualization port must be between 0 and 65535")
    return make_server(
        "127.0.0.1",
        port,
        _visualization_application(bundle),
        server_class=VisualizationServer,
        handler_class=_QuietRequestHandler,
    )
