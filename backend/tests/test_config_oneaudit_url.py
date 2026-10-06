"""config._resolve_oneaudit_base_url — Docker-only hostnames fall back to localhost on the host."""
import config


def test_unresolvable_host_falls_back_to_localhost_keeping_port_and_path():
    out = config._resolve_oneaudit_base_url("http://host.docker.internal.invalid:5030/api/v1/internal")
    assert out == "http://localhost:5030/api/v1/internal"


def test_resolvable_host_is_kept():
    assert config._resolve_oneaudit_base_url("http://localhost:5030/api/v1/internal") == "http://localhost:5030/api/v1/internal"
    assert config._resolve_oneaudit_base_url("http://127.0.0.1:5030/x") == "http://127.0.0.1:5030/x"
