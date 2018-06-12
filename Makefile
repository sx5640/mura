.POSIX:
.PHONY: help clean deps build test run run-docker

help:
	@echo 'Targets:'
	@echo '    help                Display this message.'
	@echo '    clean               Remove all build and testing artifacts.'
	@echo '    deps                Download and install all needed build dependencies.'
	@echo '    build               Build and package the software.'
	@echo '    test                Run all unit test for this repo.'
	@echo '                        manager-sesrecsys PR#3 in Stash.'
	@echo ''
	@echo '    run-docker-build    Build the Dockerfile.'
	@echo '    run-docker          Run Docker with no command attached'
	@echo '    run                 Run Docker with no command attached'
	@echo ''

clean:
	find . -iname '*.pyc' -exec rm {} +
	docker system prune -f
	docker image prune -af

deps:
	pip install --upgrade pip
	pip install -r requirements.txt

build: deps

test: deps

#---------------------------------------------------------------------------------------------------

run-docker-build: Dockerfile
    # Remove version if exit, so docker build won't leave previous version unnamed
	docker build -t personal/mura .

run-docker: run-docker-build
	docker run --rm -it

#---------------------------------------------------------------------------------------------------
run: build

