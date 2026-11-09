def test_public_api_imports():
    import ai_inference_lab as lab

    for name in lab.__all__:
        assert hasattr(lab, name), name
