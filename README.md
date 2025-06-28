## Table of Contents

- [Introduction](#introduction)
- [Learning Outcomes](#learning-outcomes)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Task One — Managing API Keys with LiteLLM Proxy](./tasks/task_1.md)
  - [Task Two — API Endpoint](./tasks/task_2.md)
  - [Task Three — Refactor to Cloud Services](./tasks/task_3.md)
  - [Task Four — Production-ready Docker Image](./tasks/task_4.md)
  - [Task Five — Deployment to AWS](./tasks/task_5.md)
- [Deliverables](#deliverables)
- [Useful Resources](#useful-resources)
- [Contributing](#contributing)
- [The Flow](#the-flow)

## **Introduction**

There are various ways of deploying LLM applications depending on your needs: for pre-trained models, hosted APIs (e.g., OpenAI) offer instant setup and minimal infrastructure. If you are deploying your own fine-tuned models, you would need a GPU-powered server (either cloud-based or local) to serve your inference endpoints.

As you know, frameworks like LangChain help you orchestrate your LLM applications, and others like FastAPI let you expose clean, RESTful API endpoints. Typically, you package your code in a Docker image and use Kubernetes or a serverless platform (like AWS Lambda) to run it, enabling automatic scaling with demand. In this task, this is what you will work on.

You already have an LLM app that is prepared for production. You did monitoring, evaluation, refined your prompts, and even added guardrails. Now, you will add a more robust API key and budgeting strategy, an API endpoint, containerize the app, and deploy it to AWS. This way, your app will be ready for everyone else to use. 

The starter code (from the final stage of LLM Evals: Let's Talk Money project) is located in the `starter_code` directory of this repo.

## **Learning Outcomes**

We cover deploying LLM applications in a production-ready environment. You'll use tools and services like LiteLLM for API key management, FastAPI for creating REST API endpoints, Docker for containerization, SaaS providers for various services, and AWS Cloud to ship and share what you've been building.

## **Project Structure**

Here are the main directories and files in this repo:

```markdown
├── images/
│   ├── litellm_dashboard.png
    ├── ...
│   └── ecr_push_commands.png
├── starter_code/
├── tasks/
│   ├── task_1.md
│   ├── task_2.md
│   ├── task_3.md
│   ├── task_4.md
│   └── task_5.md
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── main.py
├── README.md
└── requirements.txt
```

## **Tasks**

The project is divided into various tasks that you need to complete. The tasks are located in the tasks folder of the repository. Each task includes all the necessary objectives, suggested development steps, deliverables, and useful resources. Here's a brief overview of each task:

- **Task One — Managing API Keys and Budgets via LiteLLM Proxy:** Use LiteLLM to manage your API keys, set budgets per user per key, enforce rate limits, and more.
- **Task Two — API Endpoint:** Create a REST API endpoint with FastAPI that handles `POST` requests and returns a response from the LLM.
- **Task Three — Refactor to Cloud Services**: Use SaaS providers for your vector store, Redis, and Langfuse, instead of self-hosting them.
- **Task Four  — Production-ready Docker Image**: Containerize the FastAPI application using Docker to ensure it runs consistently across different environments.
- **Task Five — Deployment to AWS**: Deploy the application to AWS using AWS EC2 Instances.

## **Useful Resources**

Each task contains a collection of resources that will be helpful for you as you solve the task. There are links to topics in Hyperskill, documentation, and other helpful tutorials that you can use. You may not always need to use all the provided resources if you're already familiar with the concepts. In addition to the provided resources, you can always discuss with your teammates and experts. You can use various channels — GitHub Issues, GitHub Discussions, PRs, or Discord.

## **Deliverables**

Each task contains a set of deliverables that bring you close to achieving the final goal. The final product is an LLM application that is accessible from anywhere. Additionally, a Docker image for your application in a public container registry of your choice (you can still delete it afterward). Add the link to the image in the About section of the repo found on the right side of the repo page. 

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