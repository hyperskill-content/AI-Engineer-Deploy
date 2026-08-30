## Table of Contents

- [Introduction](#introduction)
- [Learning Outcomes](#learning-outcomes)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Task One — API Endpoint](tasks/task_1.md)
  - [Task Two — Refactor to Cloud Services](tasks/task_2.md)
  - [Task Three — Production-ready Docker Image](tasks/task_3.md)
  - [Task Four — Deployment to AWS](tasks/task_4.md)
- [Deliverables](#deliverables)
- [Useful Resources](#useful-resources)
- [Contributing](#contributing)
- [The Flow](#the-flow)

## **Introduction**

There are various ways of deploying LLM applications depending on your needs: for pre-trained models, hosted APIs (e.g., OpenAI) offer instant setup and minimal infrastructure. If you are deploying your own fine-tuned models, you would need a GPU-powered server (either cloud-based or local) to serve your inference endpoints.

As you know, frameworks like LangChain help you orchestrate your LLM applications, and others like FastAPI let you expose clean, RESTful API endpoints. Typically, you package your code in a Docker image and use Kubernetes or a serverless platform (like AWS Lambda) to run it, enabling automatic scaling with demand. In this task, this is what you will work on.

You already have an LLM app that is prepared for production. You did monitoring, evaluation, refined your prompts, cost management, and even added guardrails. Now, you will add an API endpoint, containerize the app, and deploy it to AWS. This way, your app will be ready for everyone else to use.

---

## **Setup**

This project uses [uv](https://docs.astral.sh/uv/) for Python package management. Follow these steps to get started:

### Prerequisites

1. Install uv:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Verify installation:
   ```bash
   uv --version
   ```

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AI-Engineer-Deploy/starter_code
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

   This will:
   - Create a virtual environment in `.venv/`
   - Install all dependencies from `pyproject.toml`
   - Generate/update the `uv.lock` file for reproducible builds

3. Set up environment variables:
   ```bash
   cp .env.sample .env
   # Edit .env with your API keys and configuration
   ```

### Running the Application

```bash
uv run python main.py
```

If your IDE (VS Code, PyCharm) isn't recognizing the environment or you need to use tools that aren't uv-aware, activate the virtual environment manually:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Run the application
python main.py
```

### Adding New Dependencies

To add new packages to the project:

```bash
uv add <package-name>
```

For development dependencies:

```bash
uv add --dev <package-name>
```

### Managing Environment Variables

When adding new environment variables to your project:

1. **Always update `.env.sample`** with the new variable names (without actual values):
   ```bash
   # Example: Adding a new service
   NEW_SERVICE_API_KEY="<your-api-key-here>"
   NEW_SERVICE_BASE_URL="<service-url>"
   ```

2. **Do not remove `.env.sample`** - This file serves as a template for other developers and documents all required environment variables for the project.

3. **Add descriptive comments** in `.env.sample` to explain what each variable is used for, especially if it's not immediately obvious.

### Documenting Code Structure Changes

If you modify the default application structure or change how the application runs:

1. **Update this README** with the new run instructions in the "Running the Application" section above.

2. **Document any new directories or files** in the "Project Structure" section.

3. **Explain the reasoning** for structural changes in your Pull Request description so reviewers understand the architectural decisions.

Examples of changes that require documentation updates:
- Moving code from `main.py` into modular files under `src/`
- Introducing new entry points (e.g., `app.py`, `cli.py`)
- Adding configuration files or changing how configuration is loaded
- Creating new directories for components, utilities, or services

## **Learning Outcomes**

We cover deploying LLM applications in a production-ready environment. You'll use tools and services like FastAPI for creating REST API endpoints, Docker for containerization, SaaS providers for various services, and AWS Cloud to ship and share what you've been building.

## **Project Structure**

Here are the main directories and files in this repo:

```markdown
├── images
│   ├── AWS.png
│   ├── langfuse.png
│   ├── litellm.png
│   ├── qdrant.png
│   ├── redis.png
│   └── UI.png
├── app
│   ├── config
│   │   ├── config.yml
│   │   └── prompts.yml
│   ├── datasets
│   │   └── smartphones.json
│   ├── .env.sample
│   ├── main.py
│   ├── pyproject.toml
│   └── uv.lock
├── tasks
│   ├── task_1.md
│   ├── task_2.md
│   ├── task_3.md
│   └── task_4.md
├── CONTRIBUTING.md
└── README.md
```

## **Tasks**

The project is divided into various tasks that you need to complete. The tasks are located in the tasks folder of the repository. Each task includes all the necessary objectives, suggested development steps, deliverables, and useful resources. Here's a brief overview of each task:

- **Task One — API Endpoint:** Create a REST API endpoint with FastAPI that handles `POST` requests and returns a response from the LLM.
- **Task Two — Refactor to Cloud Services**: Use SaaS providers for your vector store, Redis, and Langfuse, instead of self-hosting them.
- **Task Three  — Production-ready Docker Image**: Containerize the FastAPI application using Docker to ensure it runs consistently across different environments.
- **Task Four — Deployment to AWS**: Deploy the application to AWS using AWS EC2 Instances.

This project starts from a solution stored in the `/starter_code` directory. The tasks of this project should be completed based on the provided code.
## **Useful Resources**

Each task contains a collection of resources that will be helpful for you as you solve the task. There are links to topics in Hyperskill, documentation, and other helpful tutorials that you can use. You may not always need to use all the provided resources if you're already familiar with the concepts. In addition to the provided resources, you can always discuss with your teammates and experts. You can use various channels—GitHub Issues, GitHub Discussions, PRs, or Discord.

## **Deliverables**

Each task contains a set of deliverables that bring you close to achieving the final goal. The final product is an LLM application that is accessible from anywhere, and a Docker image for your application in a public container registry of your choice (you can still delete it afterward). Add the link to the image in the About section of the repo found on the right side of the repo page. 

## **Contributing**

All discussions and bug reports must be done via GitHub Issues, while code review is done via GitHub Pull Requests. For more information, see the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## **The Flow**
Fork → Clone → Branch → Implement → PR → Review

* Fork this repo to your own GitHub account
* Create a new branch for each task (e.g., task-1) if applicable (if there is any code that has to be implemented)
* Implement the solution based on the markdown descriptions
* Push the branch to the forked repo
* Create a Pull Request from the fork back to the main repo
* We will review the PR and provide feedback through GitHub
