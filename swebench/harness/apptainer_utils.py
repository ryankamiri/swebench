"""
Apptainer utilities to replace Docker utilities.

This module provides Apptainer-specific implementations of the functions
originally designed for Docker containers.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tarfile
import threading
import time
import traceback
from pathlib import Path

from swebench.harness.apptainer_client import ApptainerContainer, ApptainerClient

HEREDOC_DELIMITER = "EOF_1399519320"  # different from dataset HEREDOC_DELIMITERs!


def copy_to_container(container: ApptainerContainer, src: Path, dst: Path):
    """
    Copy a file from local to an Apptainer container

    Args:
        container (ApptainerContainer): Apptainer container to copy to
        src (Path): Source file path
        dst (Path): Destination file path in the container
    """
    # Check if destination path is valid
    if os.path.dirname(dst) == "":
        raise ValueError(
            f"Destination path parent directory cannot be empty!, dst: {dst}"
        )

    # For Apptainer, we'll use a different approach since containers are more ephemeral
    # We'll create a temporary directory and copy files there
    temp_dir = Path("/tmp") / f"apptainer_copy_{container.id}"
    
    try:
        # Create temporary directory in the container
        container.exec_run(f"mkdir -p {temp_dir}")
        
        # Copy file to temporary location
        container.exec_run(f"cp {src} {temp_dir / dst.name}")
        
        # Move to final destination
        container.exec_run(f"mkdir -p {dst.parent}")
        container.exec_run(f"mv {temp_dir / dst.name} {dst}")
        
        # Clean up temporary directory
        container.exec_run(f"rm -rf {temp_dir}")
        
    except Exception as e:
        # Clean up on error
        try:
            container.exec_run(f"rm -rf {temp_dir}")
        except:
            pass
        raise e


def write_to_container(container: ApptainerContainer, data: str, dst: Path):
    """
    Write a string to a file in an Apptainer container
    """
    # For Apptainer, we'll use echo with heredoc
    command = f"cat <<'{HEREDOC_DELIMITER}' > {dst}\n{data}\n{HEREDOC_DELIMITER}"
    container.exec_run(command)


def remove_image(client: ApptainerClient, image_id: str, logger=None):
    """
    Remove an Apptainer image by ID.

    Args:
        client (ApptainerClient): Apptainer client.
        image_id (str): Image ID.
        logger (logging.Logger): Logger to use for output. If None, print to stdout.
    """
    if not logger:
        # if logger is None, print to stdout
        log_info = print
        log_error = print
        raise_error = True
    elif logger == "quiet":
        # if logger is "quiet", don't print anything
        log_info = lambda x: None
        log_error = lambda x: None
        raise_error = True
    else:
        # if logger is a logger object, use it
        log_error = logger.info
        log_info = logger.info
        raise_error = False
    
    try:
        log_info(f"Attempting to remove image {image_id}...")
        client.images.remove(image_id, force=True)
        log_info(f"Image {image_id} removed.")
    except Exception as e:
        if "not found" in str(e).lower():
            log_info(f"Image {image_id} not found, removing has no effect.")
        else:
            if raise_error:
                raise e
            log_error(f"Failed to remove image {image_id}: {e}\n{traceback.format_exc()}")


def cleanup_container(client: ApptainerClient, container: ApptainerContainer, logger):
    """
    Stop and remove an Apptainer container.
    Performs this forcefully if the container cannot be stopped with the python API.

    Args:
        client (ApptainerClient): Apptainer client.
        container (ApptainerContainer): Container to remove.
        logger (logging.Logger): Logger to use for output. If None, print to stdout
    """
    if not container:
        return

    container_id = container.id

    if not logger:
        # if logger is None, print to stdout
        log_error = print
        log_info = print
        raise_error = True
    elif logger == "quiet":
        # if logger is "quiet", don't print anything
        log_info = lambda x: None
        log_error = lambda x: None
        raise_error = True
    else:
        # if logger is a logger object, use it
        log_error = logger.info
        log_info = logger.info
        raise_error = False

    # Attempt to stop the container
    try:
        if container:
            log_info(f"Attempting to stop container {container.name}...")
            container.stop(timeout=15)
    except Exception as e:
        log_error(
            f"Failed to stop container {container.name}: {e}. Trying to forcefully kill..."
        )
        try:
            # For Apptainer, we need to find and kill the process
            # This is a simplified implementation
            result = subprocess.run(
                ["pgrep", "-f", container_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        log_info(f"Forcefully killing container {container.name} with PID {pid}...")
                        os.kill(int(pid), signal.SIGKILL)
            else:
                log_error(f"No process found for container {container.name}")
                
        except Exception as e2:
            if raise_error:
                raise e2
            log_error(
                f"Failed to forcefully kill container {container.name}: {e2}\n"
                f"{traceback.format_exc()}"
            )

    # Attempt to remove the container
    try:
        log_info(f"Attempting to remove container {container.name}...")
        container.remove(force=True)
        log_info(f"Container {container.name} removed.")
    except Exception as e:
        if raise_error:
            raise e
        log_error(
            f"Failed to remove container {container.name}: {e}\n"
            f"{traceback.format_exc()}"
        )


def exec_run_with_timeout(container: ApptainerContainer, cmd: str, timeout: int | None = 60):
    """
    Run a command in a container with a timeout.

    Args:
        container (ApptainerContainer): Container to run the command in.
        cmd (str): Command to run.
        timeout (int): Timeout in seconds.
    """
    # Local variables to store the result of executing the command
    exec_result = b""
    exec_id = None
    exception = None
    timed_out = False

    # Wrapper function to run the command
    def run_command():
        nonlocal exec_result, exec_id, exception
        try:
            # For Apptainer, we'll use a simpler approach
            result = container.exec_run(cmd)
            exec_result = result.output
            exec_id = "exec_" + str(int(time.time()))
        except Exception as e:
            exception = e

    # Start the command in a separate thread
    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout)

    if exception:
        raise exception

    # If the thread is still alive, the command timed out
    if thread.is_alive():
        # For Apptainer, we can't easily kill individual exec processes
        # This is a limitation of the current implementation
        timed_out = True
    
    end_time = time.time()
    return exec_result.decode(), timed_out, end_time - start_time


def find_dependent_images(client: ApptainerClient, image_name: str):
    """
    Find all images that are built upon `image_name` image

    Args:
        client (ApptainerClient): Apptainer client.
        image_name (str): Name of the base image.
    """
    dependent_images = []

    # Get all local images
    all_images = client.images.list()

    # Get the ID of the base image
    try:
        base_image = client.images.get(image_name)
        base_image_id = base_image.id
    except Exception:
        print(f"Base image {image_name} not found.")
        return []

    for image in all_images:
        # Skip the base image itself
        if image.id == base_image_id:
            continue

        # For Apptainer, we can't easily determine dependencies
        # This is a simplified implementation
        if image_name in str(image.tags):
            dependent_images.append(image.tags[0] if image.tags else image.id)

    return dependent_images


def list_images(client: ApptainerClient):
    """
    List all images from the Apptainer client.
    """
    # don't use this in multi-threaded context
    return {tag for i in client.images.list(all=True) for tag in i.tags}


def clean_images(
    client: ApptainerClient, prior_images: set, cache_level: str, clean: bool
):
    """
    Clean Apptainer images based on cache level and clean flag.

    Args:
        client (ApptainerClient): Apptainer client.
        prior_images (set): Set of images that existed before the current run.
        cache (str): Cache level to use.
        clean (bool): Whether to clean; remove images that are higher in the cache hierarchy than the current
            cache level. E.g. if cache_level is set to env, remove all previously built instances images. if
            clean is false, previously built instances images will not be removed, but instance images built
            in the current run will be removed.
    """
    images = list_images(client)
    removed = 0
    print("Cleaning cached images...")
    for image_name in images:
        if should_remove(image_name, cache_level, clean, prior_images):
            try:
                remove_image(client, image_name, "quiet")
                removed += 1
            except Exception as e:
                print(f"Error removing image {image_name}: {e}")
                continue
    print(f"Removed {removed} images.")


def should_remove(image_name: str, cache_level: str, clean: bool, prior_images: set):
    """
    Determine if an image should be removed based on cache level and clean flag.
    """
    existed_before = image_name in prior_images
    if "/" in image_name:
        image_name = image_name.rsplit("/", 1)[-1]
    if image_name.startswith("sweb.base"):
        if cache_level in {"none"} and (clean or not existed_before):
            return True
    elif image_name.startswith("sweb.env"):
        if cache_level in {"none", "base"} and (clean or not existed_before):
            return True
    elif image_name.startswith("sweb.eval"):
        if cache_level in {"none", "base", "env"} and (clean or not existed_before):
            return True
    return False
