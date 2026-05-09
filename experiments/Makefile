SHELL := /bin/bash

.PHONY: setup install notebook clean

setup:
# 	Installing useful software
	apt-get install -y btop nvtop
	@command -v gh >/dev/null 2>&1 || (apt-get update && apt-get install -y gh)

#   Installing uv, python and then all the packages
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install 3.13.12
	uv python pin 3.13.12
	uv sync
	uv run python -m ipykernel install --user --name experiments-uv --display-name "Python (experiments-uv)"

#   GitHub auth + credential wiring (required on fresh hosts)
	@gh auth status >/dev/null 2>&1 || gh auth login --git-protocol https --web
	@gh auth setup-git
	@git remote set-url origin https://github.com/Sachin2911/Independence-Is-All-You-Need.git
	git config --global user.email "sachinmohan29@gmail.com"
	git config --global user.name "Sachin"
#   HuggingFace login
	@read -p "Enter HuggingFace token: " hf_token; \
	uv run hf auth login --token $$hf_token

notebook:
	uv run jupyter lab --ip 0.0.0.0 --port 8888 --no-browser

# Install the environment
install:
	uv sync

# Deleting the environment
clean:
	rm -rf .venv __pycache__ src/__pycache__