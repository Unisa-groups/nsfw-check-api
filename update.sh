podman-compose build
sudo systemctl stop podman-nsfw-check.service
sudo systemctl start podman-nsfw-check.service
echo done