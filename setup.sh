sudo cp podman-nsfw-check.service /etc/systemd/system/podman-nsfw-check.service
sudo systemctl daemon-reload
sudo systemctl enable podman-nsfw-check.service
sudo systemctl start podman-nsfw-check.service
sudo systemctl status podman-nsfw-check.service
