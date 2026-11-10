from aistudio_gpu.legacy_env import apply


def test_old_variables_count_as_their_new_names_unless_set():
    env = {"MLXCOMFY_IDLE": "600", "MLXCOMFY_GPU_WAN": "A100", "AISTUDIO_GPU_WAN": "H100", "OTHER": "1"}
    apply(env)
    assert env["AISTUDIO_IDLE"] == "600"
    assert env["AISTUDIO_GPU_WAN"] == "H100"
    assert "AISTUDIO_OTHER" not in env
