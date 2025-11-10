"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import zipfile
import tarfile
import gzip
import requests
from pathlib import Path
from typing import Optional, Union
import urllib.parse

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()



def download_and_extract_data(
    source: str,
    destination: str,
    extract_to: Optional[str] = None,
    remove_source: bool = True,
    force_download: bool = False
) -> Path:
    """
    Downloads data from a source URL and extracts it if compressed.
    
    Args:
        source (str): URL to download data from
        destination (str): Directory to save downloaded file
        extract_to (str, optional): Directory to extract files to (defaults to destination)
        remove_source (bool): Whether to remove the source file after extraction
        force_download (bool): Whether to force download even if file exists
    
    Returns:
        Path: Path to the extracted data directory
        
    Example usage:
        # Download and extract zip file
        data_path = download_and_extract_data(
            source="https://example.com/data.zip",
            destination="downloaded_data"
        )
        
        # Download and extract tar.gz file
        data_path = download_and_extract_data(
            source="https://example.com/data.tar.gz",
            destination="downloaded_data"
        )
    """
    
    # Create destination directory
    data_path = Path(destination)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Get filename from URL
    parsed_url = urllib.parse.urlparse(source)
    filename = Path(parsed_url.path).name
    
    if not filename:
        filename = "downloaded_file"
    
    local_file_path = data_path / filename
    
    # Download file if it doesn't exist or force_download is True
    if not local_file_path.exists() or force_download:
        print(f"[INFO] Downloading {filename} from {source}...")
        
        try:
            response = requests.get(source, stream=True)
            response.raise_for_status()
            
            # Show progress for large files
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            print(f"\r[INFO] Download progress: {percent:.1f}%", end="")
            
            print()  # New line after progress
            print(f"[INFO] Download completed: {local_file_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to download {source}: {e}")
            raise
    
    else:
        print(f"[INFO] File already exists: {local_file_path}")
    
    # Determine extraction directory
    extract_path = Path(extract_to) if extract_to else data_path
    
    # Check if file is compressed and extract if needed
    extracted_path = extract_file(local_file_path, extract_path, remove_source)
    
    return extracted_path

def extract_file(file_path: Path, extract_to: Path, remove_source: bool = True) -> Path:
    """
    Extracts compressed files based on their extension.
    
    Args:
        file_path (Path): Path to the compressed file
        extract_to (Path): Directory to extract to
        remove_source (bool): Whether to remove source after extraction
    
    Returns:
        Path: Path to extracted content
    """
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Check if extraction is needed
    supported_extensions = ['.zip', '.tar', '.tar.gz', '.tgz', '.gz']
    file_ext = file_path.suffix.lower()
    
    if not any(file_path.name.lower().endswith(ext) for ext in supported_extensions):
        print(f"[INFO] File {file_path} is not compressed, returning file path")
        return file_path
    
    print(f"[INFO] Extracting {file_path} to {extract_to}...")
    
    try:
        # Extract based on file type
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
        elif file_path.suffix in ['.tar', '.tgz'] or file_path.suffixes[-2:] == ['.tar', '.gz']:
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
                
        elif file_path.suffix == '.gz':
            # For single .gz files (not tar.gz)
            output_path = extract_to / file_path.stem
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        print(f"[INFO] Extraction completed: {extract_to}")
        
        # Remove source file if requested
        if remove_source:
            file_path.unlink()
            print(f"[INFO] Removed source file: {file_path}")
            
    except Exception as e:
        print(f"[ERROR] Failed to extract {file_path}: {e}")
        raise
    
    return extract_to

def get_image_paths(data_directory: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> list:
    """
    Gets all image file paths from a directory and its subdirectories.
    
    Args:
        data_directory (str): Directory to search for images
        extensions (tuple): Tuple of image file extensions to include
    
    Returns:
        list: List of Path objects for all image files
    """
    data_path = Path(data_directory)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(data_path.rglob(f"*{ext}"))
        image_paths.extend(data_path.rglob(f"*{ext.upper()}"))
    
    print(f"[INFO] Found {len(image_paths)} images in {data_directory}")
    return sorted(image_paths)

# Example usage
if __name__ == "__main__":
    # Download and extract data
    data_path = download_and_extract_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="data",
        extract_to="data/pizza_steak_sushi"
    )
    
    # Get all image paths
    image_paths = get_image_paths(data_path)
    
    print(f"[INFO] First 5 image paths: {image_paths[:5]}")
    print(f"[INFO] Data ready at: {data_path}")

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names