import torch
from fastapi.testclient import TestClient

from api.app import app
from api.services.inference import InferenceService
from api.services.job_manager import JobManager, Status


def _setup_app_state(app, inference=None):
    settings_dict = {
        "inference": inference or InferenceService.__new__(InferenceService),
        "job_manager": JobManager(),
    }
    for key, val in settings_dict.items():
        setattr(app.state, key, val)


class TestHealthEndpoint:
    def test_health_no_model(self):
        _setup_app_state(app)
        app.state.inference.model = None
        app.state.inference.encoders = None
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "model_loaded": False}


class TestRecommendationsEndpoint:
    def _make_inference_service(self):
        from models.mlp import MLPRecommender

        model = MLPRecommender(
            num_users=5, num_items=10, embedding_dim=8, hidden_dims=[16, 8]
        )
        model.eval()

        service = InferenceService.__new__(InferenceService)
        service.model = model
        service.encoders = {
            "user_encoder": {"user_1": 0, "user_2": 1, "user_3": 2},
            "item_encoder": {
                "item_A": 0,
                "item_B": 1,
                "item_C": 2,
                "item_D": 3,
                "item_E": 4,
                "item_F": 5,
                "item_G": 6,
                "item_H": 7,
                "item_I": 8,
                "item_J": 9,
            },
            "num_users": 5,
            "num_items": 10,
        }
        service.device = torch.device("cpu")
        return service

    def test_recommendations_success(self):
        service = self._make_inference_service()
        _setup_app_state(app, inference=service)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/recommendations/user_1?top_k=3")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_1"
        assert len(data["recommendations"]) == 3
        assert data["recommendations"][0]["rank"] == 1
        assert data["recommendations"][1]["rank"] == 2
        assert data["recommendations"][2]["rank"] == 3

    def test_recommendations_user_not_found(self):
        service = self._make_inference_service()
        _setup_app_state(app, inference=service)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/recommendations/unknown_user")
        assert response.status_code == 404

    def test_recommendations_model_not_loaded(self):
        _setup_app_state(app)
        app.state.inference.model = None
        app.state.inference.encoders = None
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/recommendations/user_1")
        assert response.status_code == 503


class TestPipelineEndpoints:
    def test_list_jobs_empty(self):
        _setup_app_state(app)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/pipeline/jobs")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_job_not_found(self):
        _setup_app_state(app)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/pipeline/jobs/nonexistent")
        assert response.status_code == 404


class TestJobManager:
    def test_create_and_get_job(self):
        jm = JobManager()
        job_id = jm.create_job("training")
        job = jm.get_job(job_id)
        assert job is not None
        assert job.step == "training"
        assert job.status == Status.PENDING

    def test_list_jobs(self):
        jm = JobManager()
        jm.create_job("preprocessing")
        jm.create_job("training")
        assert len(jm.list_jobs()) == 2

    def test_has_running_job(self):
        jm = JobManager()
        job_id = jm.create_job("training")
        assert not jm.has_running_job("training")
        jm.run_in_background(job_id, lambda: None)
        import time

        time.sleep(0.1)
        assert not jm.has_running_job("training")

    def test_run_in_background_failure(self):
        jm = JobManager()
        job_id = jm.create_job("training")

        def failing():
            raise RuntimeError("boom")

        jm.run_in_background(job_id, failing)
        import time

        time.sleep(0.1)
        job = jm.get_job(job_id)
        assert job.status == Status.FAILED
        assert job.error == "boom"
