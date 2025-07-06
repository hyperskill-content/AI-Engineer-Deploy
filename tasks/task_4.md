## Production-ready Docker Image

## **Table of Contents**

- [Description](#description)
- [Recommended Development Steps](#recommended-development-steps)
- [Deliverables](#deliverables)
- [Useful Resources](#useful-resources)
    - [Docs](#docs)

### Description

We are now preparing the application for the final stages of deployment to AWS EC2 instances. Since we are using the instance solely for this application, we may not need a Docker image. However, Docker is the industry-standard and highly recommended practice. With Docker, you ensure that your application can consistently run in any environment. This avoids the age-old issue, “it works on my machine”.

### **Recommended Development Steps**

In this task, let's work on the Dockerfile to build the image. One key thing to ensure is that the image is production-ready and does not include unnecessary files. The first step to achieving that is by including a `.dockerignore` file to exclude unnecessary files (e.g.  `.git`, `__pycache__`, `.env`) to keep the build context small. It also avoids copying development artifacts into the image. These are typically the same files you’d include in your `.gitignore` file.

Next, ensure to pin specific package versions in your `requirements.txt` to ensure reproducibility. Also, use a minimal base image such as `python:3.13-slim`. However, remember to rebuild images regularly with updated base images to include security patches.

Since we don’t want the resulting image to be too large, we need to be careful about how we build the image. Here are a few things you can do:

1. Combine related commands and remove caches in the same step. For example:

    ```bash
    RUN apt-get update && \
        rm -rf /var/lib/apt/lists/*
    ```

2. Use multi-stage builds — since we have packages with compiled dependencies, you can use a builder stage and a smaller runtime stage. In the builder stage, you would install build tools (e.g `build-essential`, `libffi-dev`, `libssl-dev`). Then, create a virtual environment and install packages without caching:

    ```bash
    RUN pip install --no-cache-dir -r requirements.txt
    ```

   At this point, if you are not using a multi-stage build, you can remove all packages:

    ```bash
    RUN apt-get purge -y --auto-remove libffi-dev libssl-dev build-essential && \
        apt-get clean && rm -rf /var/lib/apt/lists/*
    ```

   But if you are using a multi-stage build approach, you only need to copy the virtual environment you created:

    ```bash
    COPY --from=builder /opt/venv /opt/venv
    ```


This ensures only the needed packages are in the final image. The next thing you need to do is ensure that you’re following some recommended security best practices. For example, ensure that the application runs as a non-root user. You can do this by creating a user to run the app inside the container. Another thing is passing environment variables at runtime (ensure you’re accessing variables via `os.environ` in your code):

```bash
docker run --env-file env.env -p 8000:8000 custom-image:latest
```

The final thing to consider is how your run the web server. In our code, we are running it with `uvicorn` as follows:

```python
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

In our production environment, the recommended way is to use Gunicorn’s process manager. Therefore, you can set your image’s entry point as follows:

```bash
CMD ["gunicorn",
     "--worker-class", "uvicorn.workers.UvicornWorker",
     "--bind", "0.0.0.0:8000",
     "--workers", "2",        
     "--preload",      
     "main:app"
]
```

This `CMD` tells Docker to start Gunicorn using Uvicorn’s async worker class (max 4 workers), binding the FastAPI app (`main:app`) to all network interfaces on port 8000 and preloading the application code before forking worker processes for faster spin-up. Now, you can build the image:

```bash
docker build -t hypersite:latest .
```

Great job. Now, your image is production-ready. At this point, run the image to ensure that it is working as expected. For our application, the image should be around ~600-700MB.

### **Deliverables**

At this point, you should have a `Dockerfile` that can be used to build a production-ready image to run your application. Ensure you’re ignoring unnecessary files, optimizing image size, following security best practices, and using the right production environment `CMD`. You must also ensure that the image works well in your local environment.

### **Useful Resources**

### **Docs**
- [Docker optimization](https://docs.docker.com/build-cloud/optimization/)