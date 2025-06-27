## **Introduction**

Let’s continue from where you left off in LLM Evals: Let's Talk Money, after you added input rails. Now, we'll deploy the app so that everyone else can use it. 

## **Learning Outcomes**

We cover deploying LLM applications in a production-ready environment. You'll use tools and services like LiteLLM for API key management, FastAPI for the REST API endpoints, Docker for containerization, SaaS providers for various services, and AWS Cloud to ship and share the app you've been building.

## **Project Structure**

Here are the main starter directories and files in this repo:

```
├── tasks/
│   ├── task_1.md
│   ├── task_2.md
│   ├── task_3.md
│   ├── task_4.md
│   └── task_5.md
├── images/
├── README.md
├── .gitignore
└── .env.example
```

## **Tasks**

This project is divided into various tasks that you need to complete. The tasks are located in the tasks folder of the repository. Each task includes all the necessary objectives, suggested development steps, expected outcomes, and useful resources. Here's a brief overview of each task:

- **Task One — Managing API Keys and Budgets via LiteLLM Proxy —** Use LiteLLM to manage your API keys, set budgets per user per key, rate limits, and more.
- **Task Two — API Endpoint**: Create a REST API endpoint with FastAPI that accepts user input and returns a response from the LLM.
- **Task Three — Production-ready Docker Image**: Containerize the FastAPI application using Docker to ensure it runs consistently across different environments.
- **Task Four  — Refactor to Cloud Services**: Use SaaS providers for your vector store, Redis, and Langfuse instead of self-hosting them.
- **Task Five — Deployment to AWS**: Deploy the application and LiteLLM proxy to AWS using AWS EC2 Instances.

## **Useful Resources**

Each task will contain a collection of resources that will be helpful for you as you solve the task. There are links to topics in Hyperskill, documentation, and other helpful tutorials that you and your team can use. You may not always need to use all the provided resources if you're already familiar with the concepts. In addition to the provided resources, you can always discuss with your teammates and experts. You can use various channels — GitHub Issues, GitHub Discussions, PRs, or Discord.

## **Deliverables**

The final product is an LLM application that is accessible from anywhere. Each task contains its deliverables that bring you close to achieving the final goal. 

## **The flow**
Fork → Clone → Branch → Implement → PR → Review

* Fork this repo to your own GitHub account
* Create a new branch for each task (e.g., task-1) if applicable (if there is any code that has to be implemented)
* Implement the solution based on the markdown descriptions
* Push the branch to the forked repo
* Create a Pull Request from the fork back to the main repo
* We will review the PR and provide feedback through GitHub