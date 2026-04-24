SHELL := /bin/bash

.PHONY: setup install clean

setup:
# 	Installing useful software
	apt-get install -y btop nvtop
	@command -v gh >/dev/null 2>&1 || (apt-get update && apt-get install -y gh)

#   Installing uv, python and then all the packages
	curl -LsSf https://astral.sh/uv/install.sh | sh
	. ~/.local/bin/env && uv python install 3.13.12
	. ~/.local/bin/env && uv python pin 3.13.12
	. ~/.local/bin/env && uv sync

#   GitHub auth + credential wiring (required on fresh hosts)
	@gh auth status >/dev/null 2>&1 || gh auth login --git-protocol https --web
	@gh auth setup-git
	@git remote set-url origin https://github.com/Sachin2911/Independence-Is-All-You-Need.git
	git config --global user.email "sachinmohan29@gmail.com"
  	git config --global user.name "Sachin"
#   HuggingFace login
	@read -p "Enter HuggingFace token: " hf_token; \
	. ~/.local/bin/env && uv run hf auth login --token $$hf_token

# Install the environment
install:
	uv sync

# Deleting the environment
clean:
	rm -rf .venv __pycache__ src/__pycache__