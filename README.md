


### Client:
```
$ npm run dev
```

### Api:

```
$ uv run uvicorn api.app:app --reload  
```


### ML FLow:

```
$ mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root /mlflow/artifacts
```

