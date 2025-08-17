# **Collaborating on the Project**

## **Table of Contents**

- Prerequisites
- [Basic knowledge of Git (clone, commit, push, pull)](#basic-knowledge-of-git-clone-commit-push-pull)
- [GitHub Repository Deliverable](#deliverables)
- [Best Practices](#best-practices)

## **Prerequisites**

- A GitHub account.
- Git installed on your computer.

## **Basic knowledge of Git (clone, commit, push, pull).**

1. Create or fork this repository in your account:
- Create a new GitHub repository (e.g., deploying-llm-applications).
- Add collaborators in “Settings > Collaborators" (if using a private repo) so that we have access to it.
2. Clone the forked repo to your local machine:

```bash
git clone https://github.com/username/deploying-llm-applications.git
cd deploying-llm-applications
```

3. Create a new branch for your work — always create a new branch to work on a specific feature or task. Use descriptive prefixes for your branch names like `feature/`,  `bugfix/`,  or `refactor/`:

```bash
git checkout -b feature/add-fastapi-endpoint
```

4. In case you have uncommitted changes you don’t want to lose when creating or switching between branches, you can stash them first:

```bash
git stash
```

- You can then reapply your stashed changes:

```bash
git stash apply
```

5. Make your changes locally — implement your feature or fix, and commit your changes using meaningful commit messages:

```bash
git add .
git commit -m "Add FastAPI /ask endpoint"
```

6. Push your changes to GitHub:

```bash
git push origin feature/add-fastapi-endpoint
```

7. Open a pull request (PR). This is a request to merge your changes into the main branch.
- You can open a PR from the IDE (via [GitHub plugin](https://plugins.jetbrains.com/plugin/13115-github)) or go to GitHub and:
    - Click Compare & pull request.
    - Write a clear title and description:
        - What the PR is about — that is, what does it address?
        - How you went about solving the problem and why you chose that approach (this usually makes it easier for the reviewer to know why you did what)
        - Add labels like "enhancement", "bugfix", etc. — these specify what the PR is about at a glance, allow categorization and prioritization, and can also be used to trigger automation, such as in CI/CD pipelines.
8. Code review:
- We will:
    - Check the "Files changed" tab to see what has been modified.
    - Read the code and comment if needed.
    - Run the code locally to test its functionality.
    - Pull your Docker image and run it locally.
    - Check your app’s API endpoints.
    - If we find issues, we will leave comments on specific lines.
    - If everything looks good, we will approve the PR.

## **Best Practices**
- Use meaningful commit messages.
- Use branches for features, bug fixes, or experiments.
- Use descriptive names for branches (e.g., feature/add-dockerfile, bugfix/fix-api-endpoint).
- Keep Pull Requests small and focused.
- Don't commit .idea/ or build/ directories.
- Use .gitignore properly.