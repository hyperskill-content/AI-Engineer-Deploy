#!/bin/bash
# run this on a fresh ec2 instance (ubuntu/debian) to install docker and start the app
# usage: bash ec2_setup.sh

set -e

echo "installing docker..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# add current user to docker group so we dont need sudo
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker $USER
newgrp docker

echo "docker installed!"

# pull and run the image - replace IMAGE_NAME with your registry image
# e.g. docker.io/yourusername/hypersite:latest
# or public.ecr.aws/yourrepo/hypersite:latest
IMAGE_NAME="${1:-hypersite:latest}"

echo "pulling image: $IMAGE_NAME"
docker pull $IMAGE_NAME

# make sure the .env file exists before running
if [ ! -f .env ]; then
  echo "error: .env file not found - create one with your env vars first"
  exit 1
fi

echo "starting container..."
docker run \
  -d \
  -p 80:8000 \
  --name hypersite \
  --restart always \
  --env-file ./.env \
  $IMAGE_NAME

echo "done! app should be running at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/docs"
