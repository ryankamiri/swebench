"""
Apptainer build utilities to replace Docker build functionality.

This module provides Apptainer-specific implementations for building
and managing container images.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

from swebench.harness.apptainer_client import ApptainerClient, BuildError
from swebench.harness.apptainer_utils import remove_image
from swebench.harness.constants import (
    BASE_IMAGE_BUILD_DIR,
    DOCKER_USER,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    UTF8,
)
from swebench.harness.test_spec.test_spec import (
    get_test_specs_from_dataset,
    make_test_spec,
    TestSpec,
)
from swebench.harness.utils import ansi_escape, run_threadpool


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def setup_logger(instance_id: str, log_file: Path, mode="w", add_stdout: bool = False):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.

    If `add_stdout` is True, logs will also be sent to stdout, which can be used for
    streaming ephemeral output from Modal containers.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode, encoding=UTF8)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    if add_stdout:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"%(asctime)s - {instance_id} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    # To avoid too many open files
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def build_image(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    client: ApptainerClient,
    build_dir: Path,
    nocache: bool = False,
):
    """
    Builds an Apptainer image with the given name, setup scripts, and dockerfile.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile (will be converted to Apptainer format)
        platform (str): Platform to build the image for
        client (ApptainerClient): Apptainer client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """
    # Create build directory if it doesn't exist
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a logger for the build process
    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building Apptainer image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )
    
    # Debug: print dockerfile line by line
    logger.info("Dockerfile lines:")
    for i, line in enumerate(dockerfile.split('\n'), 1):
        logger.info(f"  Line {i}: {repr(line)}")

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    
    try:
        # Write the setup scripts to the build directory
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        # Convert Dockerfile to Apptainer definition file
        definition_file = convert_dockerfile_to_apptainer(dockerfile, build_dir)
        
        # Build the Apptainer image
        logger.info(
            f"Building Apptainer image {image_name} in {build_dir} with platform {platform}"
        )
        
        # Set up environment for Apptainer - respect existing environment variables
        env = os.environ.copy()
        
        # Get cache directory from environment or use default fallback
        cachedir = env.get('APPTAINER_CACHEDIR') or env.get('SINGULARITY_CACHEDIR')
        if not cachedir:
            # Only set default if not already configured in environment
            cachedir = str(Path("logs/build_images/apptainer_images").resolve())
            env['APPTAINER_CACHEDIR'] = cachedir
            env['SINGULARITY_CACHEDIR'] = cachedir
            logger.info(f"APPTAINER_CACHEDIR not set in environment, using default: {cachedir}")
        else:
            logger.info(f"Using APPTAINER_CACHEDIR from environment: {cachedir}")
        
        # Get temp directory from environment or use default fallback
        tmpdir = env.get('APPTAINER_TMPDIR') or env.get('SINGULARITY_TMPDIR')
        if not tmpdir:
            # Only set default if not already configured in environment
            tmpdir = str((Path.cwd() / "tmp" / "apptainer").resolve())
            env['APPTAINER_TMPDIR'] = tmpdir
            env['SINGULARITY_TMPDIR'] = tmpdir
            logger.info(f"APPTAINER_TMPDIR not set in environment, using default: {tmpdir}")
        else:
            logger.info(f"Using APPTAINER_TMPDIR from environment: {tmpdir}")
        
        # Ensure directories exist
        os.makedirs(cachedir, exist_ok=True)
        os.makedirs(tmpdir, exist_ok=True)
        
        # Output SIF file path
        sif_name = f"{image_name.replace(':', '_').replace('/', '_')}.sif"
        output_path = Path(cachedir) / sif_name
        
        # Check if already exists
        if output_path.exists() and not nocache:
            logger.info(f"Image already exists at: {output_path}")
            return
        
        # Determine build strategy based on whether this is a base or env image
        # Extract the base image from the Dockerfile
        base_image = "ubuntu:22.04"  # default
        for line in dockerfile.split('\n'):
            if line.strip().upper().startswith('FROM '):
                # Extract base image, remove --platform flag
                base_image = line.strip().split(' ', 1)[1]
                if '--platform=' in base_image:
                    parts = base_image.split()
                    base_image = ' '.join([p for p in parts if not p.startswith('--platform=')])
                base_image = base_image.strip()
                break
        
        # Check if this is building from a local SWE-bench base image
        if base_image.startswith('sweb.'):
            # This is an env image building on top of a base image
            # Use apptainer build with the definition file
            logger.info(f"Building env image on top of local base: {base_image}")
            
            # Convert Dockerfile to Apptainer definition file
            definition_file = convert_dockerfile_to_apptainer(dockerfile, build_dir)
            logger.info(f"Generated Apptainer definition file at {definition_file}")
            logger.info(f"Definition content:\n{definition_file.read_text()}")
            
            # Build using the definition file
            build_cmd = ["apptainer", "build", str(output_path), str(definition_file)]
            
            logger.info(f"Running: {' '.join(build_cmd)}")
            logger.info(f"Output path: {output_path}")
            logger.info(f"APPTAINER_TMPDIR: {tmpdir}")
            logger.info(f"APPTAINER_CACHEDIR: {cachedir}")
            
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=build_dir,  # Run in build directory so relative paths work
                timeout=1800  # 30 minute timeout for large images
            )
            
            logger.info(f"Build stdout: {result.stdout}")
            if result.stderr:
                logger.info(f"Build stderr: {result.stderr}")
            logger.info(f"Build return code: {result.returncode}")
            
            if result.returncode != 0:
                logger.error(f"Build failed!")
                raise BuildError(f"Apptainer build failed: {result.stderr}", "")
            
            logger.info(f"Successfully built image to: {output_path}")
        else:
            # This is a base image - pull directly from Docker Hub
            logger.info(f"Pulling base image from Docker Hub: docker://{base_image}")
            
            pull_cmd = ["apptainer", "pull", str(output_path), f"docker://{base_image}"]
            
            logger.info(f"Running: {' '.join(pull_cmd)}")
            logger.info(f"Output path: {output_path}")
            logger.info(f"APPTAINER_TMPDIR: {tmpdir}")
            logger.info(f"APPTAINER_CACHEDIR: {cachedir}")
            
            result = subprocess.run(
                pull_cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=1800  # 30 minute timeout for large images
            )
            
            logger.info(f"Pull stdout: {result.stdout}")
            if result.stderr:
                logger.info(f"Pull stderr: {result.stderr}")
            logger.info(f"Pull return code: {result.returncode}")
            
            if result.returncode != 0:
                logger.error(f"Pull failed!")
                raise BuildError(f"Apptainer pull failed: {result.stderr}", "")
            
            logger.info(f"Successfully pulled image to: {output_path}")
        
        
    except BuildError as e:
        logger.error(f"BuildError during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)  # functions that create loggers should close them


def convert_dockerfile_to_apptainer(dockerfile: str, build_dir: Path) -> Path:
    """
    Convert a Dockerfile to an Apptainer definition file.
    
    Args:
        dockerfile (str): Contents of the Dockerfile
        build_dir (Path): Build directory
    
    Returns:
        Path: Path to the generated Apptainer definition file
    """
    # Ensure build directory exists
    build_dir.mkdir(parents=True, exist_ok=True)
    definition_file = build_dir / "apptainer.def"
    
    # Start with the Apptainer definition header
    definition_content = "Bootstrap: docker\n"
    definition_content += "From: ubuntu:20.04\n\n"
    
    # Parse Dockerfile and convert to Apptainer format
    lines = dockerfile.strip().split('\n')
    in_run_section = False
    run_commands = []
    current_run_cmd = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.upper().startswith('FROM '):
            # Extract base image, remove Docker-specific flags
            base_image = line.split(' ', 1)[1]
            # Remove --platform flag if present
            if '--platform=' in base_image:
                parts = base_image.split()
                base_image = ' '.join([p for p in parts if not p.startswith('--platform=')])
            
            # Check if this is a local SWE-bench image (not a Docker Hub image)
            if base_image.startswith('sweb.'):
                # This is a local SWE-bench image - use localimage bootstrap
                # Find the SIF file for this base image
                cachedir = os.environ.get('APPTAINER_CACHEDIR') or os.environ.get('SINGULARITY_CACHEDIR')
                if not cachedir:
                    cachedir = str(Path("logs/build_images/apptainer_images").resolve())
                
                sif_name = f"{base_image.replace(':', '_').replace('/', '_')}.sif"
                sif_path = Path(cachedir) / sif_name
                
                definition_content = f"Bootstrap: localimage\nFrom: {sif_path}\n\n"
            else:
                # Regular Docker Hub image
                definition_content = f"Bootstrap: docker\nFrom: {base_image}\n\n"
            
        elif line.upper().startswith('RUN '):
            # Collect RUN commands
            run_cmd = line[4:].strip()
            if run_cmd.startswith('[') and run_cmd.endswith(']'):
                # Handle exec form
                import json
                try:
                    cmd_parts = json.loads(run_cmd)
                    run_cmd = ' '.join(cmd_parts)
                except:
                    pass
            
            if run_cmd.endswith('\\'):
                # Multi-line command continues
                in_run_section = True
                current_run_cmd.append(run_cmd[:-1].strip())
            else:
                if in_run_section:
                    # End of multi-line command
                    current_run_cmd.append(run_cmd.strip())
                    run_commands.append(' '.join(current_run_cmd))
                    current_run_cmd = []
                    in_run_section = False
                else:
                    # Single line command
                    run_commands.append(run_cmd)
                    
        elif line.upper().startswith('COPY ') or line.upper().startswith('ADD '):
            # Handle COPY/ADD commands
            parts = line.split()
            if len(parts) >= 3:
                src = parts[1]
                dst = parts[2]
                definition_content += f"%files\n    {src} {dst}\n\n"
                
        elif line.upper().startswith('WORKDIR '):
            # Handle WORKDIR
            workdir = line.split(' ', 1)[1]
            definition_content += f"%environment\n    export WORKDIR={workdir}\n\n"
            
        elif line.upper().startswith('USER '):
            # Handle USER
            user = line.split(' ', 1)[1]
            definition_content += f"%runscript\n    cd ${{WORKDIR:-/}}\n\n"
    
    # Debug: Log collected RUN commands
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Collected {len(run_commands)} RUN commands:")
    for i, cmd in enumerate(run_commands, 1):
        logger.info(f"  RUN {i}: {repr(cmd)}")
    
    # Add the %post section with all RUN commands
    if run_commands:
        definition_content += "%post\n"
        for cmd in run_commands:
            definition_content += f"    {cmd}\n"
        definition_content += "\n"
    
    # Write the definition file
    with open(definition_file, 'w') as f:
        f.write(definition_content)
    
    # Log the generated definition for debugging
    logger.info(f"Generated Apptainer definition file at {definition_file}")
    logger.info(f"Definition content:\n{definition_content}")
    
    return definition_file


def build_base_images(
    client: ApptainerClient,
    dataset: list,
    force_rebuild: bool = False,
    namespace: str = None,
    instance_image_tag: str = None,
    env_image_tag: str = None,
):
    """
    Builds the base images required for the dataset if they do not already exist.

    Args:
        client (ApptainerClient): Apptainer client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
    """
    # Get the base images to build from the dataset
    test_specs = get_test_specs_from_dataset(
        dataset,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }

    # Build the base images
    for image_name, (dockerfile, platform) in base_images.items():
        try:
            # Check if the base image already exists
            client.images.get(image_name)
            if force_rebuild:
                # Remove the base image if it exists and force rebuild is enabled
                remove_image(client, image_name, "quiet")
            else:
                print(f"Base image {image_name} already exists, skipping build.")
                continue
        except Exception:
            pass
        # Build the base image (if it does not exist or force rebuild is enabled)
        print(f"Building base image ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )
    print("Base images built successfully.")


def get_env_configs_to_build(
    client: ApptainerClient,
    dataset: list,
    namespace: str = None,
    instance_image_tag: str = None,
    env_image_tag: str = None,
):
    """
    Returns a dictionary of image names to build scripts and dockerfiles for environment images.
    Returns only the environment images that need to be built.

    Args:
        client (ApptainerClient): Apptainer client to use for building the images
        dataset (list): List of test specs or dataset to build images for
    """
    image_scripts = dict()
    base_images = dict()
    test_specs = get_test_specs_from_dataset(
        dataset,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )

    for test_spec in test_specs:
        # Check if the base image exists
        try:
            if test_spec.base_image_key not in base_images:
                base_images[test_spec.base_image_key] = client.images.get(
                    test_spec.base_image_key
                )
            base_image = base_images[test_spec.base_image_key]
        except Exception:
            raise Exception(
                f"Base image {test_spec.base_image_key} not found for {test_spec.env_image_key}\n."
                "Please build the base images first."
            )

        # Check if the environment image exists
        image_exists = False
        try:
            env_image = client.images.get(test_spec.env_image_key)
            image_exists = True
        except Exception:
            pass
        if not image_exists:
            # Add the environment image to the list of images to build
            image_scripts[test_spec.env_image_key] = {
                "setup_script": test_spec.setup_env_script,
                "dockerfile": test_spec.env_dockerfile,
                "platform": test_spec.platform,
            }
    return image_scripts


def build_env_images(
    client: ApptainerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    namespace: str = None,
    instance_image_tag: str = None,
    env_image_tag: str = None,
):
    """
    Builds the environment images required for the dataset if they do not already exist.

    Args:
        client (ApptainerClient): Apptainer client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Get the environment images to build from the dataset
    if force_rebuild:
        env_image_keys = {
            x.env_image_key
            for x in get_test_specs_from_dataset(
                dataset,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
            )
        }
        for key in env_image_keys:
            remove_image(client, key, "quiet")
    build_base_images(
        client, dataset, force_rebuild, namespace, instance_image_tag, env_image_tag
    )
    configs_to_build = get_env_configs_to_build(
        client, dataset, namespace, instance_image_tag, env_image_tag
    )
    if len(configs_to_build) == 0:
        print("No environment images need to be built.")
        return [], []
    print(f"Total environment images to build: {len(configs_to_build)}")

    args_list = list()
    for image_name, config in configs_to_build.items():
        args_list.append(
            (
                image_name,
                {"setup_env.sh": config["setup_script"]},
                config["dockerfile"],
                config["platform"],
                client,
                ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
            )
        )

    successful, failed = run_threadpool(build_image, args_list, max_workers)
    # Show how many images failed to build
    if len(failed) == 0:
        print("All environment images built successfully.")
    else:
        print(f"{len(failed)} environment images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_images(
    client: ApptainerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    namespace: str = None,
    tag: str = None,
    env_image_tag: str = None,
):
    """
    Builds the instance images required for the dataset if they do not already exist.

    Args:
        dataset (list): List of test specs or dataset to build images for
        client (ApptainerClient): Apptainer client to use for building the images
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Build environment images (and base images as needed) first
    test_specs = list(
        map(
            lambda x: make_test_spec(
                x,
                namespace=namespace,
                instance_image_tag=tag,
                env_image_tag=env_image_tag,
            ),
            dataset,
        )
    )
    if force_rebuild:
        for spec in test_specs:
            remove_image(client, spec.instance_image_key, "quiet")
    _, env_failed = build_env_images(client, test_specs, force_rebuild, max_workers)

    if len(env_failed) > 0:
        # Don't build images for instances that depend on failed-to-build env images
        dont_run_specs = [
            spec for spec in test_specs if spec.env_image_key in env_failed
        ]
        test_specs = [
            spec for spec in test_specs if spec.env_image_key not in env_failed
        ]
        print(
            f"Skipping {len(dont_run_specs)} instances - due to failed env image builds"
        )
    print(f"Building instance images for {len(test_specs)} instances")
    successful, failed = list(), list()

    # `logger` is set to None b/c logger is created in build_instance_image
    payloads = [(spec, client, None, False) for spec in test_specs]
    # Build the instance images
    successful, failed = run_threadpool(build_instance_image, payloads, max_workers)
    # Show how many images failed to build
    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_image(
    test_spec: TestSpec,
    client: ApptainerClient,
    logger: logging.Logger | None,
    nocache: bool,
):
    """
    Builds the instance image for the given test spec if it does not already exist.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (ApptainerClient): Apptainer client to use for building the image
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
    """
    # Set up logging for the build process
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(
        ":", "__"
    )
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    # Get the image names and dockerfile for the instance image
    image_name = test_spec.instance_image_key
    env_image_name = test_spec.env_image_key
    dockerfile = test_spec.instance_dockerfile

    # Check that the env. image the instance image is based on exists
    try:
        env_image = client.images.get(env_image_name)
    except Exception as e:
        raise BuildImageError(
            test_spec.instance_id,
            f"Environment image {env_image_name} not found for {test_spec.instance_id}",
            logger,
        ) from e
    logger.info(
        f"Environment image {env_image_name} found for {test_spec.instance_id}\n"
        f"Building instance image {image_name} for {test_spec.instance_id}"
    )

    # Check if the instance image already exists
    image_exists = False
    try:
        client.images.get(image_name)
        image_exists = True
    except Exception:
        pass

    # Build the instance image
    if not image_exists:
        build_image(
            image_name=image_name,
            setup_scripts={
                "setup_repo.sh": test_spec.install_repo_script,
            },
            dockerfile=dockerfile,
            platform=test_spec.platform,
            client=client,
            build_dir=build_dir,
            nocache=nocache,
        )
    else:
        logger.info(f"Image {image_name} already exists, skipping build.")

    if new_logger:
        close_logger(logger)


def build_container(
    test_spec: TestSpec,
    client: ApptainerClient,
    run_id: str,
    logger: logging.Logger,
    nocache: bool,
    force_rebuild: bool = False,
):
    """
    Builds the instance image for the given test spec and creates a container from the image.

    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        client (ApptainerClient): Apptainer client for building image + creating the container
        run_id (str): Run ID identifying process, used for the container name
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        force_rebuild (bool): Whether to force rebuild the image even if it already exists
    """
    # Build corresponding instance image
    if force_rebuild:
        remove_image(client, test_spec.instance_image_key, "quiet")
    if not test_spec.is_remote_image:
        build_instance_image(test_spec, client, logger, nocache)
    else:
        try:
            client.images.get(test_spec.instance_image_key)
        except Exception:
            try:
                client.images.pull(test_spec.instance_image_key)
            except Exception as e:
                raise BuildImageError(test_spec.instance_id, str(e), logger) from e

    container = None
    try:
        # Create the container
        logger.info(f"Creating container for {test_spec.instance_id}...")

        # Define arguments for running the container
        run_args = test_spec.docker_specs.get("run_args", {})
        cap_add = run_args.get("cap_add", [])

        container = client.containers.create(
            image=test_spec.instance_image_key,
            name=test_spec.get_instance_container_name(run_id),
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
            platform=test_spec.platform,
            cap_add=cap_add,
        )
        logger.info(f"Container for {test_spec.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        from swebench.harness.apptainer_utils import cleanup_container
        cleanup_container(client, container, logger)
        raise BuildImageError(test_spec.instance_id, str(e), logger) from e
