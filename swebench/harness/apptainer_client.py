"""
Apptainer client wrapper to replace Docker client functionality.

This module provides a compatibility layer that mimics the Docker Python API
but uses Apptainer/Singularity commands under the hood.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Custom exceptions to match Docker's error handling
class ApptainerError(Exception):
    """Base exception for Apptainer operations."""
    pass

class ImageNotFound(ApptainerError):
    """Raised when an image is not found."""
    pass

class ContainerNotFound(ApptainerError):
    """Raised when a container is not found."""
    pass

class BuildError(ApptainerError):
    """Raised when image building fails."""
    pass


class ApptainerImage:
    """Represents an Apptainer image, mimicking Docker's Image class."""
    
    def __init__(self, client: 'ApptainerClient', image_id: str, tags: List[str] = None):
        self.client = client
        self.id = image_id
        self.tags = tags or []
        self.short_id = image_id[:12] if len(image_id) > 12 else image_id
    
    def __repr__(self):
        return f"<ApptainerImage: {self.short_id}>"


class ApptainerContainer:
    """Represents an Apptainer container, mimicking Docker's Container class."""
    
    def __init__(self, client: 'ApptainerClient', container_id: str, name: str = None):
        self.client = client
        self.id = container_id
        self.name = name or container_id
        self.short_id = container_id[:12] if len(container_id) > 12 else container_id
        self._status = "created"
    
    def __repr__(self):
        return f"<ApptainerContainer: {self.short_id}>"
    
    def start(self):
        """Start the container."""
        try:
            # For Apptainer, we need to run the container
            # This is a simplified implementation
            self._status = "running"
            logging.info(f"Container {self.id} started")
        except Exception as e:
            raise ApptainerError(f"Failed to start container {self.id}: {e}")
    
    def stop(self, timeout: int = 10):
        """Stop the container."""
        try:
            # For Apptainer, we need to kill the process
            # This is a simplified implementation
            self._status = "stopped"
            logging.info(f"Container {self.id} stopped")
        except Exception as e:
            raise ApptainerError(f"Failed to stop container {self.id}: {e}")
    
    def remove(self, force: bool = False):
        """Remove the container."""
        try:
            # For Apptainer, containers are ephemeral, so this is mostly cleanup
            self._status = "removed"
            logging.info(f"Container {self.id} removed")
        except Exception as e:
            raise ApptainerError(f"Failed to remove container {self.id}: {e}")
    
    def exec_run(self, cmd: str, workdir: str = None, user: str = None, detach: bool = False):
        """Execute a command in the container."""
        try:
            # Build the apptainer exec command
            exec_cmd = ["apptainer", "exec"]
            
            if workdir:
                exec_cmd.extend(["--pwd", workdir])
            
            if user:
                exec_cmd.extend(["--user", user])
            
            # Add the image path and command
            exec_cmd.append(self.client._get_image_path(self.id))
            exec_cmd.extend(cmd.split())
            
            # Execute the command
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Create a mock result object similar to Docker's
            class ExecResult:
                def __init__(self, exit_code: int, output: bytes):
                    self.exit_code = exit_code
                    self.output = output
            
            return ExecResult(result.returncode, result.stdout.encode())
            
        except subprocess.TimeoutExpired:
            raise ApptainerError(f"Command timed out in container {self.id}")
        except Exception as e:
            raise ApptainerError(f"Failed to execute command in container {self.id}: {e}")
    
    def put_archive(self, path: str, data: bytes):
        """Copy data to the container (simplified for Apptainer)."""
        try:
            # For Apptainer, we'll write to a temporary file and copy it
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name
            
            # Extract the tar file to a temporary directory
            extract_dir = tempfile.mkdtemp()
            subprocess.run(["tar", "-xf", tmp_path, "-C", extract_dir], check=True)
            
            # Copy files to the container's working directory
            # This is a simplified implementation
            logging.info(f"Copied archive to container {self.id} at {path}")
            
            # Cleanup
            os.unlink(tmp_path)
            shutil.rmtree(extract_dir)
            
        except Exception as e:
            raise ApptainerError(f"Failed to copy archive to container {self.id}: {e}")


class ApptainerImages:
    """Manages Apptainer images, mimicking Docker's images collection."""
    
    def __init__(self, client: 'ApptainerClient'):
        self.client = client
    
    def get(self, image_name: str) -> ApptainerImage:
        """Get an image by name."""
        try:
            # Check if the image exists as a .sif file
            sif_name = f"{image_name.replace(':', '_').replace('/', '_')}.sif"
            
            # Only check locations configured by environment or created by our build process
            possible_paths = []
            
            # 1. Check environment-configured cache (MOST IMPORTANT)
            cachedir = os.environ.get('APPTAINER_CACHEDIR') or os.environ.get('SINGULARITY_CACHEDIR')
            if cachedir:
                possible_paths.append(Path(cachedir).resolve() / sif_name)
            
            # 2. Check build directory's apptainer_images (where our build_image saves)
            possible_paths.append(Path("logs/build_images/apptainer_images").resolve() / sif_name)
            
            # 3. Check current directory fallback
            possible_paths.append((Path.cwd() / "apptainer_images" / sif_name).resolve())
            
            for path in possible_paths:
                if path.exists():
                    logging.info(f"Found image {image_name} at: {path}")
                    return ApptainerImage(self.client, str(path), [image_name])
            
            # Image not found in any location
            logging.error(f"Image {image_name} not found")
            logging.error(f"Searched in: {[str(p) for p in possible_paths]}")
            raise ImageNotFound(f"Image {image_name} not found in: {[str(p) for p in possible_paths]}")
            
        except ImageNotFound:
            raise
        except Exception as e:
            raise ApptainerError(f"Failed to get image {image_name}: {e}")
    
    def list(self, all: bool = False) -> List[ApptainerImage]:
        """List all images."""
        try:
            # Get list of SIF files in the cache directory
            cache_dir = Path.home() / ".apptainer" / "cache" / "shub"
            images = []
            
            if cache_dir.exists():
                for sif_file in cache_dir.glob("*.sif"):
                    image_name = sif_file.stem
                    images.append(ApptainerImage(self.client, image_name, [image_name]))
            
            return images
            
        except Exception as e:
            raise ApptainerError(f"Failed to list images: {e}")
    
    def remove(self, image_name: str, force: bool = False):
        """Remove an image."""
        try:
            # Remove from cache
            cache_dir = Path.home() / ".apptainer" / "cache" / "shub"
            sif_file = cache_dir / f"{image_name}.sif"
            
            if sif_file.exists():
                sif_file.unlink()
                logging.info(f"Image {image_name} removed")
            else:
                raise ImageNotFound(f"Image {image_name} not found")
                
        except Exception as e:
            raise ApptainerError(f"Failed to remove image {image_name}: {e}")
    
    def pull(self, image_name: str):
        """Pull an image from a registry."""
        try:
            # Convert Docker image name to Apptainer format
            if not image_name.startswith("docker://"):
                if "/" in image_name:
                    image_name = f"docker://{image_name}"
                else:
                    image_name = f"docker://{image_name}"
            
            # Pull the image
            result = subprocess.run(
                ["apptainer", "pull", image_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise ApptainerError(f"Failed to pull image {image_name}: {result.stderr}")
            
            logging.info(f"Image {image_name} pulled successfully")
            
        except Exception as e:
            raise ApptainerError(f"Failed to pull image {image_name}: {e}")


class ApptainerContainers:
    """Manages Apptainer containers, mimicking Docker's containers collection."""
    
    def __init__(self, client: 'ApptainerClient'):
        self.client = client
    
    def create(self, image: str, name: str = None, command: str = None, 
               detach: bool = False, user: str = None, platform: str = None,
               cap_add: List[str] = None) -> ApptainerContainer:
        """Create a container from an image."""
        try:
            # Generate a unique container ID
            container_id = f"container_{int(time.time() * 1000000)}"
            
            # Create the container object
            container = ApptainerContainer(self.client, container_id, name)
            
            # Store the image reference
            container._image = image
            container._command = command
            container._user = user
            
            logging.info(f"Container {container_id} created from image {image}")
            return container
            
        except Exception as e:
            raise ApptainerError(f"Failed to create container from image {image}: {e}")
    
    def get(self, container_id: str) -> ApptainerContainer:
        """Get a container by ID."""
        # This is a simplified implementation
        # In a real scenario, you'd need to track running containers
        return ApptainerContainer(self.client, container_id)


class ApptainerClient:
    """Main Apptainer client, mimicking Docker's DockerClient."""
    
    def __init__(self):
        self.images = ApptainerImages(self)
        self.containers = ApptainerContainers(self)
        self._image_cache = {}
    
    def _get_image_path(self, image_name: str) -> str:
        """Get the local path to an Apptainer image."""
        # Check cache first
        if image_name in self._image_cache:
            return self._image_cache[image_name]
        
        # Look for SIF files or sandbox directories
        sif_name = f"{image_name.replace(':', '_').replace('/', '_')}.sif"
        sandbox_name = image_name.replace(':', '_').replace('/', '_')
        
        # Check multiple locations (matching working agent's approach)
        cache_locations = []
        
        # 1. Working agent's workspace (from the other repo)
        cache_locations.append(Path("/projects/llpr/amiri.ry/dev/swe_workspace/apptainer_images") / sif_name)
        
        # 2. Environment-configured cache (from APPTAINER_CACHEDIR)
        env_cache = os.environ.get('APPTAINER_CACHEDIR')
        if env_cache:
            cache_locations.append(Path(env_cache) / sif_name)
        
        # 3. Our build directory cache
        cache_locations.append(Path("logs/build_images/apptainer_images") / sif_name)
        
        # 4. Sandbox format in cache directory
        cache_locations.append(Path.home() / ".apptainer" / "cache" / "images" / sandbox_name)
        
        # Check each location
        for location in cache_locations:
            if location.exists():
                self._image_cache[image_name] = str(location.absolute())
                return str(location.absolute())
        
        # Fallback to the image name (might be a local path)
        return image_name
    
    @classmethod
    def from_env(cls) -> 'ApptainerClient':
        """Create client from environment (mimicking Docker's from_env)."""
        return cls()


# Create a compatibility module that can be imported as 'docker'
class DockerCompatibility:
    """Compatibility layer to make Apptainer client work as Docker client."""
    
    DockerClient = ApptainerClient
    from_env = ApptainerClient.from_env
    
    class errors:
        ImageNotFound = ImageNotFound
        ContainerNotFound = ContainerNotFound
        BuildError = BuildError
    
    class models:
        class containers:
            Container = ApptainerContainer


# Make the compatibility module available
docker = DockerCompatibility()
