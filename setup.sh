sudo cp podman-nsfw-check.service /etc/systemd/system/podman-nsfw-check.service
sudo systemctl daemon-reload
sudo systemctl enable podman-compose-yourproject.service
sudo systemctl start podman-compose-yourproject.service
sudo systemctl status podman-compose-yourproject.service
